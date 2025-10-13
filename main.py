import os, io, json, logging, base64, sqlite3
from datetime import datetime, timezone, time as dtime
import pytz
from telegram import Update, InputFile
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, ContextTypes, filters
import openai
import pandas as pd
import matplotlib.pyplot as plt

# config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)
else:
    cfg = {}

TELEGRAM_TOKEN = cfg.get("TELEGRAM_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
OPENAI_API_KEY = cfg.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
TIMEZONE = cfg.get("TIMEZONE", "Europe/Moscow")

if not TELEGRAM_TOKEN:
    raise RuntimeError("8184639271:AAHQkduEayU0V-ILaHVUgksfJfR0vi84f7Q)")
if OPENAI_API_KEY:
    import openai as _openai
    _openai.api_key = OPENAI_API_KEY

DB_PATH = os.path.join(os.path.dirname(__file__), "aladdin_trades.db")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("aladdin_pro")

# default sessions (MSK times)
SESSIONS = cfg.get("SESSIONS", [
    {"name": "Asia",   "start": "02:00", "end": "10:00"},
    {"name": "Europe", "start": "10:00", "end": "18:00"},
    {"name": "US",     "start": "16:30", "end": "00:30"},
    {"name": "AfterHours", "start": "00:30", "end": "02:00"}
])

# DB init
def init_db():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        chat_id INTEGER,
        pair TEXT,
        direction TEXT,
        open_time TEXT,
        open_price REAL,
        size REAL,
        leverage REAL,
        close_time TEXT,
        close_price REAL,
        pnl REAL,
        pnl_percent REAL,
        note TEXT,
        session TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        chat_id INTEGER PRIMARY KEY,
        pair TEXT,
        timeframe TEXT,
        indicators TEXT
    )
    """)
    con.commit()
    con.close()

init_db()

def parse_hm(hm):
    hh, mm = map(int, hm.split(":"))
    return dtime(hh, mm)

def get_session_for_dt(dt_local):
    t = dt_local.time()
    for s in SESSIONS:
        start = parse_hm(s["start"])
        end = parse_hm(s["end"])
        if start <= end:
            if start <= t < end:
                return s["name"]
        else:
            if t >= start or t < end:
                return s["name"]
    return "Unknown"

# helpers DB/settings
def get_settings(chat_id):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT pair, timeframe, indicators FROM settings WHERE chat_id = ?", (chat_id,))
    row = cur.fetchone()
    con.close()
    if row:
        pair, timeframe, indicators = row
        return {"pair": pair, "timeframe": timeframe, "indicators": json.loads(indicators) if indicators else {}}
    else:
        return {"pair": "BTC-USD", "timeframe": "15m", "indicators": {"EMA":[9,21], "MACD":True, "RSI":14, "OBV":True}}

def save_settings(chat_id, pair=None, timeframe=None, indicators=None):
    cur_set = get_settings(chat_id)
    pair = pair or cur_set["pair"]
    timeframe = timeframe or cur_set["timeframe"]
    indicators = indicators or cur_set["indicators"]
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""INSERT INTO settings(chat_id, pair, timeframe, indicators) VALUES(?,?,?,?)
                   ON CONFLICT(chat_id) DO UPDATE SET pair=excluded.pair, timeframe=excluded.timeframe, indicators=excluded.indicators""",
                (chat_id, pair, timeframe, json.dumps(indicators)))
  con.commit()
    con.close()

def add_trade(chat_id, pair, direction, open_price, size, leverage, note=""):
    tz = pytz.timezone(TIMEZONE)
    now_local = datetime.now(timezone.utc).astimezone(tz)
    session_name = get_session_for_dt(now_local)
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("INSERT INTO trades(chat_id, pair, direction, open_time, open_price, size, leverage, note, session) VALUES(?,?,?,?,?,?,?,?,?)",
                (chat_id, pair, direction, datetime.now(timezone.utc).isoformat(), open_price, size, leverage, note, session_name))
    con.commit()
    trade_id = cur.lastrowid
    con.close()
    return trade_id

def close_trade(trade_id, close_price):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT open_price, size, leverage, direction FROM trades WHERE id = ?", (trade_id,))
    row = cur.fetchone()
    if not row:
        con.close()
        return None
    open_price, size, leverage, direction = row
    if open_price is None or open_price == 0:
        ret = 0.0
    else:
        if direction.lower() == "long":
            ret = (close_price - open_price) / open_price
        else:
            ret = (open_price - close_price) / open_price
    pnl = ret * size * leverage
    pnl_percent = ret * 100
    cur.execute("UPDATE trades SET close_time = ?, close_price = ?, pnl = ?, pnl_percent = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), close_price, pnl, pnl_percent, trade_id))
    con.commit()
    con.close()
    return {"pnl": pnl, "pnl_percent": pnl_percent}

def compute_stats(chat_id):
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trades WHERE chat_id = ?", con, params=(chat_id,), parse_dates=["open_time","close_time"])
    con.close()
    if df.empty:
        return {"trades":0}
    tz = pytz.timezone(TIMEZONE)
    df["open_time_local"] = pd.to_datetime(df["open_time"]).dt.tz_convert(tz)
    df["close_time_local"] = pd.to_datetime(df["close_time"]).dt.tz_convert(tz)
    df["win"] = df["pnl"] > 0
    total = len(df)
    wins = int(df["win"].sum())
    winrate = wins/total if total>0 else 0
    avg_pnl = float(df["pnl"].mean())
    avg_pct = float(df["pnl_percent"].mean())
    # by session
    if "session" not in df.columns:
        df["session"] = df["open_time_local"].apply(get_session_for_dt)
    by_session = df.groupby("session")["pnl"].agg(["count","sum","mean"]).reset_index().to_dict(orient="records")
    # by hour/week/month
    df["hour"] = df["open_time_local"].dt.hour
    by_hour = df.groupby("hour")["pnl"].agg(["count","sum","mean"]).reset_index().to_dict(orient="records")
    df["week"] = df["open_time_local"].dt.isocalendar().week
    by_week = df.groupby("week")["pnl"].agg(["count","sum","mean"]).reset_index().to_dict(orient="records")
    df["month"] = df["open_time_local"].dt.month
  by_month = df.groupby("month")["pnl"].agg(["count","sum","mean"]).reset_index().to_dict(orient="records")
    longs = df[df["direction"]=="long"]
    shorts = df[df["direction"]=="short"]
    # equity curve
    df_sorted = df.sort_values("open_time")
    df_sorted["cum_pnl"] = df_sorted["pnl"].cumsum().fillna(0)
    # plot equity curve to bytes
    buf = io.BytesIO()
    plt.figure(figsize=(8,4))
    plt.plot(df_sorted["open_time_local"].dt.tz_convert(tz), df_sorted["cum_pnl"], marker='o')
    plt.title("Equity curve (cum PnL)")
    plt.xlabel("Time")
    plt.ylabel("Cum PnL")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    res = {
        "trades": int(total),
        "wins": int(wins),
        "winrate": float(winrate),
        "avg_pnl": float(avg_pnl),
        "avg_pct": float(avg_pct),
        "by_session": by_session,
        "by_hour": by_hour,
        "by_week": by_week,
        "by_month": by_month,
        "longs": {"count": int(len(longs)), "pnl_sum": float(longs["pnl"].sum()) if not longs.empty else 0.0},
        "shorts": {"count": int(len(shorts)), "pnl_sum": float(shorts["pnl"].sum()) if not shorts.empty else 0.0},
        "equity_image": buf  # bytes buffer to return
    }
    return res

# Telegram handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    save_settings(chat_id, None, None, None)
    await context.bot.send_message(chat_id, "Привет! Aladdin-bot Pro v1.1 активирован. Используй /help чтобы увидеть доступные команды.")

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    txt = (
        "/set_pair <PAIR> — задать тикер (пример: BTC-USD или SPY)\n"
        "/set_tf <TF> — задать таймфрейм (пример: 15m, 1h)\n"
        "/set_indicators <JSON> — задать индикаторы (пример: {\"EMA\":[9,21],\"RSI\":14})\n"
        "/show_settings — показать текущие настройки\n"
        "/analyze — отправь картинку с графиком, или /analyze <TICKER> для анализа по тикеру\n"
        "/open_trade <direction> <size> <leverage> [price optional] [note] — открыть сделку\n"
        "/close_trade <trade_id> <close_price> — закрыть сделку по id\n"
        "/stats — показать статистику по сделкам (включая equity curve)\n"
        "/export_trades — получить CSV экспорта всех сделок\n"
    )
    await context.bot.send_message(update.effective_chat.id, txt)

async def set_pair(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if len(context.args) < 1:
        await context.bot.send_message(chat_id, "Использование: /set_pair BTC-USD")
        return
    pair = context.args[0].upper()
    save_settings(chat_id, pair=pair)
    await context.bot.send_message(chat_id, f"Пара установлена: {pair}")

async def set_tf(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
  if len(context.args) < 1:
        await context.bot.send_message(chat_id, "Использование: /set_tf 15m")
        return
    tf = context.args[0]
    save_settings(chat_id, timeframe=tf)
    await context.bot.send_message(chat_id, f"Таймфрейм установлен: {tf}")

async def set_indicators(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if len(context.args) < 1:
        await context.bot.send_message(chat_id, "Использование: /set_indicators {\"EMA\":[9,21],\"RSI\":14}")
        return
    try:
        ind = json.loads(" ".join(context.args))
        save_settings(chat_id, indicators=ind)
        await context.bot.send_message(chat_id, f"Индикаторы обновлены: {ind}")
    except Exception as e:
        await context.bot.send_message(chat_id, "Ошибка парсинга JSON: " + str(e))

async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    s = get_settings(chat_id)
    await context.bot.send_message(chat_id, "Текущие настройки: " + json.dumps(s, ensure_ascii=False))

async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if context.args:
        ticker = context.args[0].upper()
        settings = get_settings(chat_id)
        await context.bot.send_message(chat_id, f"Анализ по тикеру {ticker} (tf={settings['timeframe']}):\n(Это краткий mock-анализ).")
        return
    else:
        await context.bot.send_message(chat_id, "Отправь изображение графика после команды /analyze или просто отправь фото напрямую.")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    msg = await context.bot.send_message(chat_id, "Принял скрин, готовлюся к анализу...")
    photo = update.message.photo[-1]
    bio = io.BytesIO()
    await photo.get_file().download(out=bio)
    b64 = base64.b64encode(bio.getvalue()).decode()
    settings = get_settings(chat_id)
    # For now, mock analysis response
    result = {"summary":"Mock analysis: no OpenAI key", "probabilities":{"down":40,"flat":30,"up":30}, "trade_block":{"entry":"market","stop":None,"take":None}, "confidence":0.4}
    txt = f"Краткий анализ (mock):\nСводка: {result.get('summary')}\nВероятности: {result.get('probabilities')}\nТрейдер-блок: {result.get('trade_block')}\nУверенность: {result.get('confidence')}"
    await context.bot.send_message(chat_id, txt)

async def open_trade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if len(context.args) < 3:
        await context.bot.send_message(chat_id, "Использование: /open_trade <long|short> <size> <leverage> [price] [note]")
        return
    direction = context.args[0].lower()
    size = float(context.args[1])
    leverage = float(context.args[2])
    price = None
    note = ""
    if len(context.args) >= 4:
        try:
          price = float(context.args[3])
        except:
            price = None
    if len(context.args) >= 5:
        note = " ".join(context.args[4:])
    settings = get_settings(chat_id)
    pair = settings.get("pair","BTC-USD")
    trade_id = add_trade(chat_id, pair, direction, price or 0.0, size, leverage, note)
    await context.bot.send_message(chat_id, f"Открыта сделка id={trade_id}, {direction} {pair}, size={size}, lev={leverage}, price={'market' if price is None else price}")

async def close_trade_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    if len(context.args) < 2:
        await context.bot.send_message(chat_id, "Использование: /close_trade <trade_id> <close_price>")
        return
    tid = int(context.args[0])
    price = float(context.args[1])
    res = close_trade(tid, price)
    if res is None:
        await context.bot.send_message(chat_id, "Сделка не найдена.")
    else:
        await context.bot.send_message(chat_id, f"Сделка {tid} закрыта. PnL={res['pnl']} ({res['pnl_percent']}%)")

async def stats_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    s = compute_stats(chat_id)
    if s.get("trades",0)==0:
        await context.bot.send_message(chat_id, "Нет сделок для статистики.")
        return
    txt = f"""Статистика торговли:
Всего сделок: {s['trades']}
Выигрышей: {s['wins']} (Winrate {s['winrate']:.2%})
Средний PnL: {s['avg_pnl']:.4f}
Средний %: {s['avg_pct']:.2f}%
Лонгов: {s['longs']['count']} (PnL sum {s['longs']['pnl_sum']:.4f})
Шортов: {s['shorts']['count']} (PnL sum {s['shorts']['pnl_sum']:.4f})
"""
    await context.bot.send_message(chat_id, txt)
    await context.bot.send_message(chat_id, "Разбивка по сессиям:")
    await context.bot.send_message(chat_id, json.dumps(s.get("by_session",[]), ensure_ascii=False, indent=2))
    # send equity curve image
    buf = s.get("equity_image")
    if buf:
        buf.seek(0)
        await context.bot.send_photo(chat_id, buf)

async def export_trades_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    con = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trades WHERE chat_id = ?", con, params=(chat_id,))
    con.close()
    if df.empty:
        await context.bot.send_message(chat_id, "Нет сделок для экспорта.")
        return
    bio = io.BytesIO()
    df.to_csv(bio, index=False)
    bio.seek(0)
    await context.bot.send_document(chat_id, InputFile(bio, filename="trades_export.csv"))

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("set_pair", set_pair))
    app.add_handler(CommandHandler("set_tf", set_tf))
    app.add_handler(CommandHandler("set_indicators", set_indicators))
    app.add_handler(CommandHandler("show_settings", show_settings))
    app.add_handler(CommandHandler("analyze", analyze_cmd))
    app.add_handler(CommandHandler("open_trade", open_trade_cmd))
    app.add_handler(CommandHandler("close_trade", close_trade_cmd))
    app.add_handler(CommandHandler("stats", stats_cmd))
    app.add_handler(CommandHandler("export_trades", export_trades_cmd))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    logger.info("Starting Aladdin-bot Pro v1.1...")
    app.run_polling()

if __name__ == "__main__":
    main()
