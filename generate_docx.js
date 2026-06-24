const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, HeadingLevel, BorderStyle, WidthType, ShadingType,
  VerticalAlign, LevelFormat, PageNumber, Footer
} = require('docx');
const fs = require('fs');

const OUT = 'C:\\Users\\santi\\LSTM_Pytorch\\LSTM_Explained.docx';

// ── Colours ───────────────────────────────────────────────────────────────────
const COL = {
  title:    '1A1A2E',
  h2:       '2E4057',
  body:     '2D2D2D',
  note:     '555555',
  code:     'C0392B',
  codeBg:   'F4F4F4',
  green:    '1A6B2A',
  tblHdr:   '2E4057',
  tblAlt:   'F0F4F8',
  white:    'FFFFFF',
  rule:     'CCCCCC',
};

// ── Run helpers ───────────────────────────────────────────────────────────────
const r   = (t, x={}) => new TextRun({ text: t, font:'Arial', size:22, color:COL.body, ...x });
const rb  = (t)        => r(t, { bold:true });
const ri  = (t)        => r(t, { italics:true, color:COL.note });
const rc  = (t)        => new TextRun({ text:t, font:'Courier New', size:19, color:COL.code });
const rg  = (t)        => r(t, { bold:true, color:COL.green, size:26 });

// ── Paragraph helpers ─────────────────────────────────────────────────────────
const body    = (ch)  => new Paragraph({ children: Array.isArray(ch)?ch:[r(ch)], spacing:{after:80} });
const note    = (t)   => new Paragraph({ children:[ri(t)], spacing:{before:40, after:80} });
const gap     = ()    => new Paragraph({ children:[r('')], spacing:{after:100} });

const h1 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_1,
  spacing: { before:280, after:120 },
  children: [new TextRun({ text:t, font:'Arial', size:34, bold:true, color:COL.title })]
});
const h2 = (t) => new Paragraph({
  heading: HeadingLevel.HEADING_2,
  spacing: { before:180, after:80 },
  children: [new TextRun({ text:t, font:'Arial', size:26, bold:true, color:COL.h2 })]
});

const bullet = (ch) => new Paragraph({
  numbering: { reference:'bullets', level:0 },
  spacing:   { after:60 },
  children:  Array.isArray(ch) ? ch : [r(ch)],
});

const hr = () => new Paragraph({
  children: [],
  border:   { bottom:{ style:BorderStyle.SINGLE, size:6, color:COL.rule } },
  spacing:  { before:140, after:140 }
});

// Code block: one paragraph per line
const codeBlock = (lines) => lines.map(line =>
  new Paragraph({
    children:  [new TextRun({ text: line==='' ? ' ' : line, font:'Courier New', size:18, color:'1E1E1E' })],
    spacing:   { before:0, after:0 },
    shading:   { fill:COL.codeBg, type:ShadingType.CLEAR },
    indent:    { left:360, right:360 },
  })
);

// ── Table helper ──────────────────────────────────────────────────────────────
const cellBorder = { style:BorderStyle.SINGLE, size:1, color:COL.rule };
const borders    = { top:cellBorder, bottom:cellBorder, left:cellBorder, right:cellBorder };

function tbl(headers, rows, colW) {
  const tw = colW.reduce((a,b)=>a+b, 0);

  const hdrRow = new TableRow({ children: headers.map((h,i) =>
    new TableCell({
      borders, width:{ size:colW[i], type:WidthType.DXA },
      shading:  { fill:COL.tblHdr, type:ShadingType.CLEAR },
      margins:  { top:80, bottom:80, left:130, right:130 },
      verticalAlign: VerticalAlign ? VerticalAlign.CENTER : undefined,
      children: [new Paragraph({ children:[new TextRun({ text:h, font:'Arial', size:19, bold:true, color:COL.white })] })]
    })
  )});

  const dataRows = rows.map((row, ri) => new TableRow({ children: row.map((cell,ci) =>
    new TableCell({
      borders, width:{ size:colW[ci], type:WidthType.DXA },
      shading:  { fill: ri%2===0 ? COL.tblAlt : COL.white, type:ShadingType.CLEAR },
      margins:  { top:80, bottom:80, left:130, right:130 },
      children: [new Paragraph({
        children: typeof cell === 'string'
          ? [new TextRun({ text:cell, font:'Arial', size:18, color:COL.body })]
          : cell
      })]
    })
  )}));

  return new Table({ width:{ size:tw, type:WidthType.DXA }, columnWidths:colW, rows:[hdrRow,...dataRows] });
}

// ── Document children ─────────────────────────────────────────────────────────
const children = [

  // ── Cover
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing:   { before:480, after:120 },
    children:  [new TextRun({ text:'How the EXC LSTM Stock Predictor Works', font:'Arial', size:44, bold:true, color:COL.title })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing:   { after:240 },
    children:  [new TextRun({ text:'From the basics of LSTMs to MC-Dropout forecasting', font:'Arial', size:24, italics:true, color:COL.h2 })]
  }),
  hr(),
  body([r('This document walks through every function in '), rc('main.py'), r(', explaining the theory behind each decision in plain language. No deep learning background required.')]),
  gap(),

  // ── 1. What is an LSTM?
  h1('1. What is an LSTM?'),
  h2('Recurrent Neural Networks (RNNs)'),
  body([r('A standard neural network processes one input at a time and has no memory. A Recurrent Neural Network (RNN) adds a loop: the output at step '), ri('t'), r(' is fed back as extra input at step '), ri('t+1'), r('. This gives the network a form of short-term memory — useful for sequences like time series, text, or audio.')]),
  note('Analogy: imagine reading a sentence word by word. Each word makes more sense when you remember what came before it. An RNN does the same with numbers.'),

  h2('The Vanishing Gradient Problem'),
  body([r('When training an RNN on long sequences (e.g. 60 days of prices), gradients must travel backwards through every time step. Each step multiplies the gradient by a small weight, so after 60 steps the gradient shrinks to nearly zero. The network stops learning anything from events far in the past — this is the '), rb('vanishing gradient problem'), r('.')]),

  h2('How LSTMs Solve It'),
  body([r('Long Short-Term Memory networks (LSTMs, Hochreiter & Schmidhuber, 1997) replace the simple RNN cell with a more complex unit containing two conveyor belts of information:')]),
  bullet([rb('Cell state (C)'), r(' — the long-term memory. It flows through the sequence with only minor, targeted modifications.')]),
  bullet([rb('Hidden state (h)'), r(' — the short-term memory / working output used at each step.')]),
  body([r('Three '), rb('gates'), r(' control what information is kept or discarded:')]),
  gap(),
  tbl(
    ['Gate', 'Question it answers', 'Effect'],
    [
      ['Forget gate', 'What old memory should I erase?',    'Multiplies cell state by values close to 0 (forget) or 1 (keep)'],
      ['Input gate',  'What new information should I store?', 'Decides which new values are written into the cell state'],
      ['Output gate', 'What should I output right now?',    'Filters the cell state to produce the current hidden state h'],
    ],
    [1661, 3471, 4228]
  ),
  gap(),
  note('Analogy: think of the cell state as a whiteboard. The forget gate decides what to erase, the input gate writes new notes, and the output gate decides what you read aloud to the next step.'),

  // ── 2. Data Pipeline
  h1('2. The Data Pipeline'),
  h2('load_and_prepare_data()'),
  body('This function downloads 15 years of daily EXC (Exelon Corporation) stock data from Yahoo Finance and engineers seven features the LSTM will learn from:'),
  gap(),
  tbl(
    ['Feature', 'What it measures', 'Why it helps'],
    [
      ['Close',   'Daily closing price (USD)',                    'The target variable — what we want to predict'],
      ['Volume',  'Shares traded that day',                       'High volume often signals conviction behind a price move'],
      ['SMA_20',  'Simple 20-day moving average',                 'Smooths noise; shows the medium-term trend direction'],
      ['EMA_20',  'Exponential 20-day moving average',            'Like SMA but weights recent days more heavily'],
      ['RSI_14',  'Relative Strength Index (0-100)',              'Above 70 = overbought, below 30 = oversold; flags reversals'],
      ['MACD',    'MACD difference (fast EMA - slow EMA - signal)','Captures momentum shifts and trend changes'],
      ['OBV',     'On-Balance Volume (cumulative)',               'Tracks whether volume is flowing into or out of the stock'],
    ],
    [1283, 3093, 4984]
  ),
  gap(),

  h2('MinMaxScaler — Normalisation'),
  body([r('Raw values vary wildly: Close is ~$40, Volume is in the millions, OBV can be in the billions. Neural networks learn poorly from such mismatched scales. '), rc('MinMaxScaler'), r(' rescales every feature to the range [0, 1]:')]),
  ...codeBlock([
    'scaler = MinMaxScaler()',
    'scaler.fit(values[:train_end])   # <-- only on training data',
    'scaled = scaler.transform(values)',
  ]),
  gap(),

  h2('Why fit only on training data?'),
  body([r('If you fit the scaler on the entire dataset (including future prices), the scaler\'s min/max encodes information about future prices into the training normalization. The model would unknowingly "see" the future. This is called '), rb('data leakage'), r(' and produces metrics that look great but fail completely on real future data.')]),
  note('Fix: fit the scaler only on the training slice (70% of the data), then use those same statistics to transform the validation and test sets.'),

  h2('make_sequences() — The Sliding Window'),
  body([r('LSTMs expect sequences, not individual rows. This function creates a '), rb('60-day rolling window'), r(': for every day in the dataset it packages the previous 60 days as one input sample, with the next day\'s Close as the target.')]),
  ...codeBlock([
    '# For i = 60, 61, 62 ...',
    'X[i] = scaled[i-60 : i]   # shape: (60, 7) — 60 days x 7 features',
    'y[i] = scaled[i, 0]        # column 0 = normalised Close',
  ]),
  gap(),
  note('Why 60 days? It covers roughly three trading months — long enough to capture medium-term trends and indicator patterns without being so long that the network forgets the beginning of the window.'),

  // ── 3. Model
  h1('3. The Model: LSTMRegressor'),
  body([r('The model class inherits from '), rc('nn.Module'), r(' (PyTorch\'s base class for all neural networks). It has three layers:')]),
  ...codeBlock([
    'class LSTMRegressor(nn.Module):',
    '    def __init__(self, input_dim, hidden_dim, num_layers, dropout):',
    '        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,',
    '                            batch_first=True, dropout=dropout)',
    '        self.drop = nn.Dropout(dropout)',
    '        self.fc   = nn.Linear(hidden_dim, 1)',
    '',
    '    def forward(self, x):',
    '        out, _ = self.lstm(x)           # process all 60 time steps',
    '        return self.fc(self.drop(out[:, -1, :]))  # predict from last step',
  ]),
  gap(),

  h2('nn.LSTM — the core'),
  bullet([rb('input_dim = 7'),    r(' — one value per feature at each time step.')]),
  bullet([rb('hidden_dim = 64'),  r(' — each LSTM cell outputs a 64-dimensional vector (its hidden state).')]),
  bullet([rb('num_layers = 2'),   r(' — two LSTM layers stacked. The second layer receives the hidden states of the first, letting it learn higher-level patterns.')]),
  bullet([rb('batch_first=True'), r(' — input shape is '), rc('(batch, 60, 7)'), r(' instead of '), rc('(60, batch, 7)'), r('.')]),
  gap(),

  h2('nn.Dropout — Regularisation'),
  body('During training, dropout randomly sets 20% of neuron outputs to zero at each forward pass. This forces the network to learn redundant representations — no single neuron can be relied upon too heavily. The result is a model that generalises better to unseen data rather than memorising the training set.'),
  note('Analogy: it\'s like studying for an exam while randomly covering parts of your notes. You end up understanding the material deeply rather than memorising specific sentences.'),

  h2('nn.Linear — The Output Head'),
  body([r('After processing 60 time steps, we take only the '), rb('last hidden state'), r(' ('), rc('out[:, -1, :]'), r(' — shape: '), rc('(batch, 64)'), r('). This vector summarises everything the LSTM learned from the 60-day window. The linear layer compresses it from 64 dimensions down to 1 — the predicted normalised closing price.')]),

  // ── 4. Training
  h1('4. Training: the train() function'),
  h2('Loss Function — MSELoss'),
  body([rc('nn.MSELoss'), r(' computes the '), rb('Mean Squared Error'), r(' between predictions and true values. Squaring the errors penalises large mistakes more heavily, pushing the model to avoid big mispredictions.')]),
  bullet([r('MSE = mean((predicted − actual)²')]),
  gap(),

  h2('Optimiser — Adam'),
  body('Adam (Adaptive Moment Estimation) adjusts the learning rate individually for each parameter based on the history of gradients. It combines momentum (memory of past gradients) with RMSprop (adaptive step sizes). In practice it converges faster and more reliably than plain SGD for most deep learning tasks.'),

  h2('Gradient Clipping'),
  ...codeBlock(['nn.utils.clip_grad_norm_(model.parameters(), 1.0)']),
  gap(),
  body('During backpropagation through 60 time steps, gradients can occasionally explode — growing so large they corrupt the weights in a single step. Clipping caps the global gradient norm at 1.0, preventing this while leaving normal-sized gradients untouched.'),

  h2('Learning Rate Scheduler — ReduceLROnPlateau'),
  body([r('If validation loss does not improve for '), rc('LR_PATIENCE=5'), r(' epochs in a row, the learning rate is halved. This lets the model take large steps early in training and smaller, more precise steps later (fine-tuning without overshooting).')]),

  h2('Early Stopping'),
  body([r('At every epoch we check whether validation loss improved. If it does, we save a copy of the weights ('), rc('best_weights'), r('). If it fails to improve for '), rc('ES_PATIENCE=15'), r(' consecutive epochs, training stops and the best weights are restored. This prevents '), rb('overfitting'), r(' — training too long makes the model memorise the training data and perform worse on new data.')]),
  ...codeBlock([
    'if val_loss < best_val:',
    '    best_weights = {k: v.clone() for k, v in model.state_dict().items()}',
    '    no_improve = 0',
    'else:',
    '    no_improve += 1',
    '    if no_improve >= ES_PATIENCE:',
    '        break',
    '',
    'model.load_state_dict(best_weights)  # restore the best checkpoint',
  ]),
  gap(),
  note('In our run: training stopped at epoch 45 with best validation loss at epoch 30. Without early stopping, the model would have overfit for another 55 epochs.'),

  // ── 5. Evaluation
  h1('5. Evaluation: evaluate() and _inverse_close()'),
  h2('The Inverse Transform Problem'),
  body([r('The model predicts normalised Close values (between 0 and 1). To get real USD prices we need to invert the scaler. But '), rc('scaler.inverse_transform()'), r(' expects all 7 features — it does not know how to invert just one column. '), rc('_inverse_close()'), r(' works around this by building a dummy matrix of zeros with the predicted values placed in column 0, inverting the whole matrix, then extracting just column 0.')]),
  ...codeBlock([
    'def _inverse_close(scaled_1d, scaler, n_feat):',
    '    buf = np.zeros((len(scaled_1d), n_feat))  # dummy matrix of zeros',
    '    buf[:, 0] = scaled_1d                      # place Close in column 0',
    '    return scaler.inverse_transform(buf)[:, 0] # invert & extract column 0',
  ]),
  gap(),

  h2('Metrics Explained'),
  gap(),
  tbl(
    ['Metric', 'Formula', 'Plain English', 'Our result'],
    [
      ['RMSE', 'sqrt(mean((pred-true)^2))', 'Typical error in USD. Penalises large errors heavily.', '$1.14'],
      ['MAE',  'mean(|pred-true|)',          'Average absolute error in USD. More robust to outliers.', '$0.80'],
      ['MAPE', 'mean(|pred-true|/true)x100', 'Average % error. How wrong are we on average?',          '1.87%'],
    ],
    [1080, 2231, 4740, 1309]
  ),
  gap(),
  body([r('A MAPE of '), rb('1.87%'), r(' means the model\'s predictions are off by less than 2 cents for every dollar of stock price on average — strong performance for a single-asset LSTM.')]),

  // ── 6. MC-Dropout Forecast
  h1('6. Forecasting with Uncertainty: mc_forecast()'),
  h2('Why Standard Forecasts Have No Uncertainty'),
  body([r('A normal neural network in inference mode ('), rc('model.eval()'), r(') produces a single deterministic prediction. It gives you one number with no indication of how confident it is. For financial forecasting this is dangerous — a 30-day projection with no error bars is false precision.')]),

  h2('MC-Dropout: Uncertainty for Free'),
  body([r('MC-Dropout (Gal & Ghahramani, 2016) exploits a key insight: if you keep dropout '), rb('active at inference time'), r(' (by calling '), rc('model.train()'), r(' instead of '), rc('model.eval()'), r('), every forward pass randomly deactivates different neurons, producing a '), rb('different prediction'), r(' each time. Running 100 such passes gives you a '), rb('distribution'), r(' over predictions:')]),
  bullet([rb('Mean'), r(' of 100 runs = the forecast.')]),
  bullet([rb('Std'), r(' of 100 runs = the uncertainty band (MC ±1σ).')]),
  gap(),
  ...codeBlock([
    'model.train()   # dropout stays ON during inference',
    '',
    'runs = []',
    'for _ in range(MC_SAMPLES):   # 100 stochastic forward passes',
    '    preds = roll_forward_30_days(model, seq)',
    '    runs.append(preds)',
    '',
    'future_mean = np.array(runs).mean(axis=0)  # central forecast',
    'future_std  = np.array(runs).std(axis=0)   # uncertainty per day',
  ]),
  gap(),
  note('Analogy: instead of asking one person to predict tomorrow\'s weather, you ask 100 slightly different versions of the same forecaster. Their agreement gives confidence; their disagreement reveals uncertainty.'),

  h2('_recompute_last_indicators() — Live Feature Updates'),
  body([r('A naive forecast would freeze the technical indicators (SMA, EMA, RSI, MACD) at their last known values and just shift the window. This is wrong — if the predicted price rises sharply, RSI should reflect that overbought condition. '), rc('_recompute_last_indicators()'), r(' solves this by rebuilding all indicators from the growing price history after each predicted step:')]),
  ...codeBlock([
    'for step in range(FORECAST_DAYS):',
    '    next_close = model.predict(current_window)   # predict next price',
    '',
    '    close_buf.append(next_close)                  # extend price history',
    '    sma, ema, rsi, macd = _recompute_last_indicators(close_buf, vol_buf)',
    '    obv += vol if next_close > prev_close else -vol  # incremental OBV',
    '',
    '    new_row = [next_close, volume, sma, ema, rsi, macd, obv]',
    '    current_window = slide_window(new_row)        # shift window by 1 day',
  ]),
  gap(),
  body([r('OBV is updated '), rb('incrementally'), r(' rather than recomputing the full cumulative sum each step — this avoids an O(n) operation inside the inner loop of 100 × 30 = 3,000 iterations.')]),

  // ── 7. Results
  h1('7. Results'),
  h2('Data & Training Setup'),
  gap(),
  tbl(
    ['Setting', 'Value'],
    [
      ['Ticker',         'EXC (Exelon Corporation)'],
      ['History',        '15 years of daily OHLCV data'],
      ['Features',       '7  (Close, Volume, SMA20, EMA20, RSI14, MACD, OBV)'],
      ['Split',          '70% train / 15% val / 15% test  (no leakage)'],
      ['Window',         '60 trading days'],
      ['Model',          '2-layer LSTM, hidden=64, dropout=0.2  (52,033 params)'],
      ['Optimiser',      'Adam lr=0.001, gradient clip=1.0'],
      ['Scheduler',      'ReduceLROnPlateau, patience=5, factor=0.5'],
      ['Early stopping', 'Patience=15 epochs  ->  stopped at epoch 45'],
      ['Uncertainty',    'MC-Dropout, 100 samples x 30 days'],
    ],
    [2566, 6794]
  ),
  gap(),

  h2('Test Set Metrics'),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing:   { after:120 },
    children:  [rg('RMSE = $1.14   |   MAE = $0.80   |   MAPE = 1.87%')]
  }),
  body([r('These results mean the model\'s test-set predictions deviate from real prices by about '), rb('1.87% on average'), r('. For a stock trading around $40, that is roughly a $0.75 average error — well within a single day\'s typical price swing.')]),

  h2('Important Caveats'),
  bullet([rb('Past performance does not guarantee future results.'), r(' Financial markets are non-stationary; regime changes (recessions, policy shifts) can invalidate historical patterns.')]),
  bullet([r('The model predicts '), rb('price level'), r(', not '), rb('direction'), r('. A low MAPE does not mean the model correctly predicts whether the stock goes up or down.')]),
  bullet([r('The uncertainty band (MC ±1σ) grows with time as errors compound over 30 days — treat distant forecasts with more scepticism.')]),
  gap(),
  hr(),
  body([ri('Generated from main.py — LSTM_Pytorch project. Model weights saved to EXC_lstm.pt.')]),
];

// ── Assemble document ─────────────────────────────────────────────────────────
const doc = new Document({
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{
        level: 0,
        format: LevelFormat.BULLET,
        text: '•',
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left:720, hanging:360 } } }
      }]
    }]
  },
  styles: {
    default: { document: { run: { font:'Arial', size:22, color:COL.body } } }
  },
  sections: [{
    properties: {
      page: {
        size:   { width:12240, height:15840 },
        margin: { top:1440, right:1440, bottom:1440, left:1440 }
      }
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text:'Page ', font:'Arial', size:18, color:'888888' }),
            new TextRun({ children:[PageNumber.CURRENT], font:'Arial', size:18, color:'888888' }),
          ]
        })]
      })
    },
    children
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log('Saved:', OUT);
}).catch(err => {
  console.error(err.message);
  process.exit(1);
});
