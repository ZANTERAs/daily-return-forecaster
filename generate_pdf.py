from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Preformatted,
    HRFlowable, Table, TableStyle, KeepTogether
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER

OUTPUT = r"C:\Users\santi\LSTM_Pytorch\LSTM_Explained.pdf"

# ── Styles ────────────────────────────────────────────────────────────────────

base = getSampleStyleSheet()

title_style = ParagraphStyle("DocTitle",
    fontSize=22, leading=28, spaceAfter=6, alignment=TA_CENTER,
    textColor=colors.HexColor("#1a1a2e"), fontName="Helvetica-Bold")

subtitle_style = ParagraphStyle("DocSubtitle",
    fontSize=12, leading=16, spaceAfter=20, alignment=TA_CENTER,
    textColor=colors.HexColor("#4a4a6a"), fontName="Helvetica")

h1 = ParagraphStyle("H1",
    fontSize=16, leading=20, spaceBefore=18, spaceAfter=6,
    textColor=colors.HexColor("#1a1a2e"), fontName="Helvetica-Bold",
    borderPad=4)

h2 = ParagraphStyle("H2",
    fontSize=13, leading=17, spaceBefore=12, spaceAfter=4,
    textColor=colors.HexColor("#2e4057"), fontName="Helvetica-Bold")

body = ParagraphStyle("Body",
    fontSize=10, leading=15, spaceAfter=6,
    textColor=colors.HexColor("#2d2d2d"), fontName="Helvetica")

bullet = ParagraphStyle("Bullet",
    fontSize=10, leading=15, spaceAfter=4, leftIndent=16,
    textColor=colors.HexColor("#2d2d2d"), fontName="Helvetica",
    bulletIndent=6)

note = ParagraphStyle("Note",
    fontSize=9, leading=13, spaceAfter=6, leftIndent=12,
    textColor=colors.HexColor("#555555"), fontName="Helvetica-Oblique")

code_style = ParagraphStyle("Code",
    fontSize=8.5, leading=13, spaceAfter=2,
    textColor=colors.HexColor("#1e1e1e"), fontName="Courier",
    backColor=colors.HexColor("#f4f4f4"), leftIndent=12, rightIndent=12,
    borderPad=6)

result_style = ParagraphStyle("Result",
    fontSize=10, leading=14, spaceAfter=4, alignment=TA_CENTER,
    textColor=colors.HexColor("#006400"), fontName="Helvetica-Bold")


def B(text):
    return f"<b>{text}</b>"

def I(text):
    return f"<i>{text}</i>"

def C(text):
    return f'<font name="Courier" size="9" color="#c0392b">{text}</font>'

def HR():
    return HRFlowable(width="100%", thickness=1,
                      color=colors.HexColor("#cccccc"), spaceAfter=10, spaceBefore=4)

def code_block(text):
    return [
        Spacer(1, 4),
        Preformatted(text, code_style),
        Spacer(1, 6),
    ]

def section(title, number):
    return [
        Spacer(1, 10),
        HR(),
        Paragraph(f"{number}. {title}", h1),
        Spacer(1, 4),
    ]


# ── Document ──────────────────────────────────────────────────────────────────

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    leftMargin=0.85 * inch,
    rightMargin=0.85 * inch,
    topMargin=0.9 * inch,
    bottomMargin=0.9 * inch,
)

story = []

# ── Cover ─────────────────────────────────────────────────────────────────────

story += [
    Spacer(1, 30),
    Paragraph("How the EXC LSTM Stock Predictor Works", title_style),
    Paragraph("From the basics of LSTMs to MC-Dropout forecasting", subtitle_style),
    HR(),
    Spacer(1, 6),
    Paragraph(
        "This document walks through every function in <b>main.py</b>, explaining the "
        "theory behind each decision in plain language. No deep learning background required.",
        body),
    Spacer(1, 16),
]

# ── 1. What is an LSTM? ───────────────────────────────────────────────────────

story += section("What is an LSTM?", 1)

story += [
    Paragraph(B("Recurrent Neural Networks (RNNs)"), h2),
    Paragraph(
        "A standard neural network processes one input at a time and has no memory. "
        "A Recurrent Neural Network (RNN) adds a loop: the output at step "
        + I("t") + " is fed back as extra input at step " + I("t+1") + ". "
        "This gives the network a form of short-term memory — useful for sequences "
        "like time series, text, or audio.",
        body),
    Paragraph(
        I("Analogy: imagine reading a sentence word by word. Each word makes more sense "
          "when you remember what came before it. An RNN does the same with numbers."),
        note),

    Paragraph(B("The Vanishing Gradient Problem"), h2),
    Paragraph(
        "When training an RNN on long sequences (e.g. 60 days of prices), gradients must "
        "travel backwards through every time step. Each step multiplies the gradient by a "
        "small weight, so after 60 steps the gradient shrinks to nearly zero. "
        "The network stops learning anything from events far in the past — "
        "this is the " + B("vanishing gradient problem") + ".",
        body),

    Paragraph(B("How LSTMs Solve It"), h2),
    Paragraph(
        "Long Short-Term Memory networks (LSTMs, Hochreiter & Schmidhuber, 1997) replace "
        "the simple RNN cell with a more complex unit containing two conveyor belts of information:",
        body),
    Paragraph(u"•  " + B("Cell state (C)") + " — the long-term memory. It flows through "
              "the sequence with only minor, targeted modifications.", bullet),
    Paragraph(u"•  " + B("Hidden state (h)") + " — the short-term memory / working "
              "output used at each step.", bullet),
    Spacer(1, 6),
    Paragraph("Three " + B("gates") + " control what information is kept or discarded:", body),
]

gate_data = [
    [B("Gate"), B("Question it answers"), B("Effect")],
    ["Forget gate", "What old memory should I erase?",
     "Multiplies cell state by values close to 0 (forget) or 1 (keep)"],
    ["Input gate", "What new information should I store?",
     "Decides which new values are written into the cell state"],
    ["Output gate", "What should I output right now?",
     "Filters the cell state to produce the current hidden state h"],
]
gate_table = Table(gate_data, colWidths=[1.1*inch, 2.3*inch, 2.8*inch])
gate_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2e4057")),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f0f4f8"), colors.white]),
    ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
]))
story += [Spacer(1, 6), gate_table, Spacer(1, 6)]

story += [
    Paragraph(
        I("Analogy: think of the cell state as a whiteboard. The forget gate decides "
          "what to erase, the input gate writes new notes, and the output gate decides "
          "what you read aloud to the next step."),
        note),
]

# ── 2. Data Pipeline ──────────────────────────────────────────────────────────

story += section("The Data Pipeline", 2)

story += [
    Paragraph(B("load_and_prepare_data()"), h2),
    Paragraph(
        "This function downloads 15 years of daily EXC (Exelon Corporation) stock data "
        "from Yahoo Finance and engineers seven features the LSTM will learn from:",
        body),
]

feat_data = [
    [B("Feature"), B("What it measures"), B("Why it helps")],
    ["Close",   "Daily closing price (USD)",
     "The target variable — what we want to predict"],
    ["Volume",  "Shares traded that day",
     "High volume often signals conviction behind a price move"],
    ["SMA_20",  "Simple 20-day moving average",
     "Smooths noise; shows the medium-term trend direction"],
    ["EMA_20",  "Exponential 20-day moving average",
     "Like SMA but weights recent days more heavily"],
    ["RSI_14",  "Relative Strength Index (0-100)",
     "Above 70 = overbought, below 30 = oversold; flags reversals"],
    ["MACD",    "MACD difference (fast EMA - slow EMA - signal)",
     "Captures momentum shifts and trend changes"],
    ["OBV",     "On-Balance Volume (cumulative)",
     "Tracks whether volume is flowing into or out of the stock"],
]
feat_table = Table(feat_data, colWidths=[0.85*inch, 2.05*inch, 3.3*inch])
feat_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2e4057")),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f0f4f8"), colors.white]),
    ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
]))
story += [Spacer(1, 4), feat_table, Spacer(1, 8)]

story += [
    Paragraph(B("MinMaxScaler — Normalisation"), h2),
    Paragraph(
        "Raw values vary wildly: Close is ~$40, Volume is in the millions, "
        "OBV can be in the billions. Neural networks learn poorly from such mismatched scales. "
        + C("MinMaxScaler") + " rescales every feature to the range [0, 1]:",
        body),
] + code_block(
    "scaler = MinMaxScaler()\n"
    "scaler.fit(values[:train_end])   # <-- only on training data\n"
    "scaled = scaler.transform(values)"
) + [
    Paragraph(B("Why fit only on training data?"), h2),
    Paragraph(
        "If you fit the scaler on the entire dataset (including future prices), "
        "the scaler's min/max would encode information about future prices into the "
        "training normalization. The model would unknowingly 'see' the future. "
        "This is called " + B("data leakage") + " and produces metrics that look great "
        "but fail completely on real future data.",
        body),
    Paragraph(
        I("Fix: fit the scaler only on the training slice (70% of the data), "
          "then use those same statistics to transform the validation and test sets."),
        note),

    Paragraph(B("make_sequences() — The Sliding Window"), h2),
    Paragraph(
        "LSTMs expect sequences, not individual rows. This function creates a "
        + B("60-day rolling window") + ": for every day in the dataset it packages "
        "the previous 60 days as one input sample, with the next day's Close as the target.",
        body),
] + code_block(
    "# For i = 60, 61, 62 ...\n"
    "X[i] = scaled[i-60 : i]   # shape: (60, 7) — 60 days x 7 features\n"
    "y[i] = scaled[i, 0]        # column 0 = normalised Close"
) + [
    Paragraph(
        "Why 60 days? It covers roughly three trading months — long enough to capture "
        "medium-term trends and indicator patterns without being so long that the "
        "network forgets the beginning of the window.",
        note),
]

# ── 3. The Model ──────────────────────────────────────────────────────────────

story += section("The Model: LSTMRegressor", 3)

story += [
    Paragraph(
        "The model class inherits from " + C("nn.Module") + " (PyTorch's base class for "
        "all neural networks). It has three layers:",
        body),
] + code_block(
    "class LSTMRegressor(nn.Module):\n"
    "    def __init__(self, input_dim, hidden_dim, num_layers, dropout):\n"
    "        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,\n"
    "                            batch_first=True, dropout=dropout)\n"
    "        self.drop = nn.Dropout(dropout)\n"
    "        self.fc   = nn.Linear(hidden_dim, 1)\n"
    "\n"
    "    def forward(self, x):\n"
    "        out, _ = self.lstm(x)           # process all 60 time steps\n"
    "        return self.fc(self.drop(out[:, -1, :]))  # predict from last step"
) + [
    Paragraph(B("nn.LSTM — the core"), h2),
    Paragraph(u"•  " + B("input_dim = 7") + " — one value per feature at each time step.", bullet),
    Paragraph(u"•  " + B("hidden_dim = 64") + " — each LSTM cell outputs a 64-dimensional "
              "vector (its hidden state).", bullet),
    Paragraph(u"•  " + B("num_layers = 2") + " — two LSTM layers stacked. The second layer "
              "receives the hidden states of the first as its input, letting it learn "
              "higher-level patterns.", bullet),
    Paragraph(u"•  " + B("batch_first=True") + " — input shape is "
              + C("(batch, 60, 7)") + " instead of " + C("(60, batch, 7)") + ".", bullet),
    Spacer(1, 6),

    Paragraph(B("nn.Dropout — Regularisation"), h2),
    Paragraph(
        "During training, dropout randomly sets 20% of neuron outputs to zero at each "
        "forward pass. This forces the network to learn redundant representations — "
        "no single neuron can be relied upon too heavily. The result is a model that "
        "generalises better to unseen data rather than memorising the training set.",
        body),
    Paragraph(
        I("Analogy: it's like studying for an exam while randomly covering parts of "
          "your notes. You end up understanding the material deeply rather than "
          "memorising specific sentences."),
        note),

    Paragraph(B("nn.Linear — The Output Head"), h2),
    Paragraph(
        "After processing 60 time steps, we take only the " + B("last hidden state") +
        " (" + C("out[:, -1, :]") + " — shape: " + C("(batch, 64)") + "). "
        "This vector summarises everything the LSTM learned from the 60-day window. "
        "The linear layer compresses it from 64 dimensions down to 1 — the predicted "
        "normalised closing price.",
        body),
]

# ── 4. Training ───────────────────────────────────────────────────────────────

story += section("Training: the train() function", 4)

story += [
    Paragraph(B("Loss Function — MSELoss"), h2),
    Paragraph(
        C("nn.MSELoss") + " computes the " + B("Mean Squared Error") + " between "
        "predictions and true values. Squaring the errors penalises large mistakes "
        "more heavily than small ones, pushing the model to avoid big mispredictions.",
        body),
    Paragraph(u"•  MSE = mean((predicted - actual)<super>2</super>)", bullet),
    Spacer(1, 4),

    Paragraph(B("Optimiser — Adam"), h2),
    Paragraph(
        "Adam (Adaptive Moment Estimation) adjusts the learning rate individually for "
        "each parameter based on the history of gradients. It combines momentum (memory "
        "of past gradients) with RMSprop (adaptive step sizes). In practice it converges "
        "faster and more reliably than plain SGD for most deep learning tasks.",
        body),

    Paragraph(B("Gradient Clipping"), h2),
] + code_block(
    "nn.utils.clip_grad_norm_(model.parameters(), 1.0)"
) + [
    Paragraph(
        "During backpropagation through 60 time steps, gradients can occasionally "
        "explode — growing so large they corrupt the weights in a single step. "
        "Clipping caps the global gradient norm at 1.0, preventing this while "
        "leaving normal-sized gradients untouched.",
        body),

    Paragraph(B("Learning Rate Scheduler — ReduceLROnPlateau"), h2),
    Paragraph(
        "If validation loss does not improve for " + C("LR_PATIENCE=5") + " epochs in a row, "
        "the learning rate is halved. This lets the model take large steps early in "
        "training (fast progress) and smaller, more precise steps later "
        "(fine-tuning without overshooting).",
        body),

    Paragraph(B("Early Stopping"), h2),
    Paragraph(
        "At every epoch we check whether validation loss improved. If it does, we save "
        "a copy of the weights (" + C("best_weights") + "). If it fails to improve for "
        + C("ES_PATIENCE=15") + " consecutive epochs, training stops and the best weights "
        "are restored. This prevents " + B("overfitting") + " — training too long makes "
        "the model memorise the training data and perform worse on new data.",
        body),
] + code_block(
    "if val_loss < best_val:\n"
    "    best_weights = {k: v.clone() for k, v in model.state_dict().items()}\n"
    "    no_improve = 0\n"
    "else:\n"
    "    no_improve += 1\n"
    "    if no_improve >= ES_PATIENCE:\n"
    "        break\n"
    "\n"
    "model.load_state_dict(best_weights)  # restore the best checkpoint"
) + [
    Paragraph(
        I("In our run: training stopped at epoch 45 with best validation loss at epoch 30. "
          "Without early stopping, the model would have overfit for another 55 epochs."),
        note),
]

# ── 5. Evaluation ─────────────────────────────────────────────────────────────

story += section("Evaluation: evaluate() and _inverse_close()", 5)

story += [
    Paragraph(B("The Inverse Transform Problem"), h2),
    Paragraph(
        "The model predicts normalised Close values (between 0 and 1). To get real USD "
        "prices we need to invert the scaler. But " + C("scaler.inverse_transform()") +
        " expects all 7 features — it doesn't know how to invert just one column. "
        + C("_inverse_close()") + " works around this by building a dummy matrix of zeros "
        "with the predicted values placed in column 0 (the Close column), inverting the "
        "whole matrix, then extracting just column 0.",
        body),
] + code_block(
    "def _inverse_close(scaled_1d, scaler, n_feat):\n"
    "    buf = np.zeros((len(scaled_1d), n_feat))  # dummy matrix of zeros\n"
    "    buf[:, 0] = scaled_1d                      # place Close in column 0\n"
    "    return scaler.inverse_transform(buf)[:, 0] # invert & extract column 0"
) + [
    Paragraph(B("Metrics Explained"), h2),
]

metric_data = [
    [B("Metric"), B("Formula"), B("Plain English"), B("Our result")],
    ["RMSE", "sqrt(mean((pred-true)^2))",
     "Typical error in USD. Penalises large errors heavily.", "$1.14"],
    ["MAE",  "mean(|pred-true|)",
     "Average absolute error in USD. More robust to outliers.", "$0.80"],
    ["MAPE", "mean(|pred-true|/true) x100",
     "Average % error. Intuitive: how wrong are we on average?", "1.87%"],
]
metric_table = Table(metric_data, colWidths=[0.75*inch, 1.55*inch, 2.6*inch, 0.9*inch])
metric_table.setStyle(TableStyle([
    ("BACKGROUND",    (0, 0), (-1, 0),  colors.HexColor("#2e4057")),
    ("TEXTCOLOR",     (0, 0), (-1, 0),  colors.white),
    ("FONTNAME",      (0, 0), (-1, 0),  "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS",(0, 1), (-1, -1), [colors.HexColor("#f0f4f8"), colors.white]),
    ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("VALIGN",        (0, 0), (-1, -1), "TOP"),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 6),
]))
story += [Spacer(1, 4), metric_table, Spacer(1, 6)]

story += [
    Paragraph(
        "A MAPE of " + B("1.87%") + " means the model's predictions are off by less than 2 cents "
        "for every dollar of stock price on average — strong performance for a single-asset LSTM.",
        note),
]

# ── 6. MC-Dropout Forecast ────────────────────────────────────────────────────

story += section("Forecasting with Uncertainty: mc_forecast()", 6)

story += [
    Paragraph(B("Why Standard Forecasts Have No Uncertainty"), h2),
    Paragraph(
        "A normal neural network in inference mode (" + C("model.eval()") + ") produces "
        "a single deterministic prediction. It gives you one number with no indication "
        "of how confident it is. For financial forecasting this is dangerous — a "
        "30-day projection with no error bars is false precision.",
        body),

    Paragraph(B("MC-Dropout: Uncertainty for Free"), h2),
    Paragraph(
        "MC-Dropout (Gal & Ghahramani, 2016) exploits a key insight: "
        "if you keep dropout " + B("active at inference time") + " (by calling "
        + C("model.train()") + " instead of " + C("model.eval()") + "), every forward "
        "pass randomly deactivates different neurons, producing a " + B("different prediction") +
        " each time. Running 100 such passes gives you a " + B("distribution") + " over predictions:",
        body),
    Paragraph(u"•  " + B("Mean") + " of 100 runs = the forecast.", bullet),
    Paragraph(u"•  " + B("Std") + " of 100 runs = the uncertainty band (MC +/-1 sigma).", bullet),
    Spacer(1, 4),
] + code_block(
    "model.train()   # dropout stays ON during inference\n"
    "\n"
    "runs = []\n"
    "for _ in range(MC_SAMPLES):   # 100 stochastic forward passes\n"
    "    preds = roll_forward_30_days(model, seq)\n"
    "    runs.append(preds)\n"
    "\n"
    "future_mean = np.array(runs).mean(axis=0)  # central forecast\n"
    "future_std  = np.array(runs).std(axis=0)   # uncertainty per day"
) + [
    Paragraph(
        I("Analogy: instead of asking one person to predict tomorrow's weather, you ask "
          "100 slightly different versions of the same forecaster. Their agreement gives "
          "confidence; their disagreement reveals uncertainty."),
        note),

    Paragraph(B("_recompute_last_indicators() — Live Feature Updates"), h2),
    Paragraph(
        "A naive forecast would freeze the technical indicators (SMA, EMA, RSI, MACD) "
        "at their last known values and just shift the window. This is wrong — if the "
        "predicted price rises sharply, RSI should reflect that overbought condition. "
        + C("_recompute_last_indicators()") + " solves this by rebuilding all indicators "
        "from the growing price history after each predicted step:",
        body),
] + code_block(
    "for step in range(FORECAST_DAYS):\n"
    "    next_close = model.predict(current_window)   # predict next price\n"
    "\n"
    "    close_buf.append(next_close)                  # extend price history\n"
    "    sma, ema, rsi, macd = _recompute_last_indicators(close_buf, volume_buf)\n"
    "    obv += volume if next_close > prev_close else -volume  # incremental OBV\n"
    "\n"
    "    new_row = [next_close, volume, sma, ema, rsi, macd, obv]\n"
    "    current_window = slide_window(new_row)        # shift window by 1 day"
) + [
    Paragraph(
        "OBV is updated " + B("incrementally") + " rather than recomputing the full "
        "cumulative sum each step — this avoids an O(n) operation inside the inner loop "
        "of 100 x 30 = 3,000 iterations.",
        note),
]

# ── 7. Results ────────────────────────────────────────────────────────────────

story += section("Results", 7)

story += [
    Paragraph(B("Data & Training Setup"), h2),
]

setup_data = [
    ["Ticker",           "EXC (Exelon Corporation)"],
    ["History",          "15 years of daily OHLCV data"],
    ["Features",         "7  (Close, Volume, SMA20, EMA20, RSI14, MACD, OBV)"],
    ["Split",            "70% train / 15% val / 15% test  (no leakage)"],
    ["Window",           "60 trading days"],
    ["Model",            "2-layer LSTM, hidden=64, dropout=0.2  (52,033 params)"],
    ["Optimiser",        "Adam lr=0.001, gradient clip=1.0"],
    ["Scheduler",        "ReduceLROnPlateau, patience=5, factor=0.5"],
    ["Early stopping",   "Patience=15 epochs  ->  stopped at epoch 45"],
    ["Uncertainty",      "MC-Dropout, 100 samples x 30 days"],
]
setup_table = Table(setup_data, colWidths=[1.7*inch, 4.5*inch])
setup_table.setStyle(TableStyle([
    ("FONTNAME",      (0, 0), (0, -1), "Helvetica-Bold"),
    ("FONTSIZE",      (0, 0), (-1, -1), 9),
    ("ROWBACKGROUNDS",(0, 0), (-1, -1), [colors.HexColor("#f0f4f8"), colors.white]),
    ("GRID",          (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
    ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ("LEFTPADDING",   (0, 0), (-1, -1), 8),
]))
story += [Spacer(1, 4), setup_table, Spacer(1, 12)]

story += [
    Paragraph(B("Test Set Metrics"), h2),
    Spacer(1, 4),
    Paragraph("RMSE = $1.14   |   MAE = $0.80   |   MAPE = 1.87%", result_style),
    Spacer(1, 8),
    Paragraph(
        "These results mean the model's test-set predictions deviate from real prices "
        "by about " + B("1.87% on average") + ". For a stock trading around $40, that is "
        "roughly an $0.75 average error — well within a single day's typical price swing.",
        body),

    Paragraph(B("Important Caveats"), h2),
    Paragraph(u"•  " + B("Past performance does not guarantee future results.") +
              " Financial markets are non-stationary; regime changes (recessions, policy "
              "shifts) can invalidate historical patterns.", bullet),
    Paragraph(u"•  The model predicts " + B("price level") + ", not " + B("direction") +
              ". A low MAPE does not mean the model correctly predicts whether the stock "
              "goes up or down.", bullet),
    Paragraph(u"•  The uncertainty band (MC +/-1 sigma) grows with time as errors "
              "compound over 30 days — treat distant forecasts with more scepticism.", bullet),
    Spacer(1, 12),
    HR(),
    Paragraph(
        "Generated from <b>main.py</b> — LSTM_Pytorch project. "
        "Model weights saved to <b>EXC_lstm.pt</b>.",
        note),
]

# ── Build ─────────────────────────────────────────────────────────────────────

doc.build(story)
print(f"PDF saved to: {OUTPUT}")
