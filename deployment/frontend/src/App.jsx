import React, { useEffect, useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const REFERENCE_FALLBACK = {
  benchmark: {
    id: 'abcan_m515',
    name: 'abCAN M515',
    target: 'Antibody–antigen ΔΔG (kcal/mol)',
  },
  published_reference: {
    rmse_kcal_mol: 1.46,
    pcc: 0.731,
    table_rmse_kcal_mol: 1.476,
    source: 'https://academic.oup.com/bib/article/26/5/bbaf464/8251565',
    note: 'The article reports two RMSE values for this benchmark.',
  },
  current_project_result: null,
  project_status: 'awaiting_original_abcan_training_data',
};

const futureDatasets = [
  {
    name: 'AbAgym',
    description: 'Assay-specific deep mutational scanning benchmark',
    status: 'Coming soon',
  },
  {
    name: 'SAbDab2',
    description: 'Structure-focused evaluation',
    status: 'Coming soon',
  },
];

const futureApproaches = [
  { name: 'Sequence-only', description: 'Sequence model comparison' },
  { name: 'Structure-only', description: 'Interface graph comparison' },
  { name: 'Ensemble', description: 'Combined model comparison' },
];

function MetricCard({ label, value, unit, tone = 'default', caption }) {
  return (
    <article className={'metric-card metric-card-' + tone}>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">
        {value}
        {unit && <span className="metric-unit">{unit}</span>}
      </strong>
      <span className="metric-caption">{caption}</span>
    </article>
  );
}

export default function App() {
  const [benchmark, setBenchmark] = useState(REFERENCE_FALLBACK);
  const [apiStatus, setApiStatus] = useState('checking');
  const [predictionReady, setPredictionReady] = useState(false);
  const [pdbFile, setPdbFile] = useState(null);
  const [mutation, setMutation] = useState({
    chain: 'A',
    resnum: 45,
    wild: 'K',
    mut: 'R',
  });
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState('');

  useEffect(() => {
    axios.get(API_BASE + '/benchmark')
      .then((response) => setBenchmark({ ...REFERENCE_FALLBACK, ...response.data }))
      .catch(() => setApiStatus('offline'));

    axios.get(API_BASE + '/health')
      .then((response) => {
        setApiStatus('online');
        setPredictionReady(Boolean(response.data.prediction_ready));
      })
      .catch(() => setApiStatus('offline'));
  }, []);

  const reference = benchmark.published_reference || REFERENCE_FALLBACK.published_reference;
  const current = benchmark.current_project_result;
  const activeBenchmark = benchmark.active_benchmark || REFERENCE_FALLBACK.benchmark;
  const projectStatus = benchmark.project_status || REFERENCE_FALLBACK.project_status;
  const projectStatusLabel = projectStatus.replace(/_/g, ' ');

  const handleSubmit = async (event) => {
    event.preventDefault();
    setResult(null);
    setErrorMessage('');

    if (!predictionReady) {
      setErrorMessage('Predictions will be enabled after the abCAN M515 model is trained and validated.');
      return;
    }
    if (!pdbFile) {
      setErrorMessage('Choose a PDB structure first.');
      return;
    }

    const formData = new FormData();
    formData.append('pdb_file', pdbFile);
    formData.append('chain', mutation.chain);
    formData.append('resnum', mutation.resnum);
    formData.append('wild_aa', mutation.wild);
    formData.append('mut_aa', mutation.mut);

    try {
      const response = await axios.post(API_BASE + '/predict/single', formData);
      setResult(response.data);
    } catch (error) {
      setErrorMessage(
        error.response?.data?.detail || 'The abCAN prediction service is unavailable.'
      );
    }
  };

  const connectionLabel = apiStatus === 'online'
    ? 'API connected'
    : apiStatus === 'offline'
      ? 'API offline'
      : 'Connecting';

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="abCAN home">
          <span className="brand-mark">a</span>
          <span>abCAN<span className="brand-light"> / benchmark</span></span>
        </a>
        <div className={'service-status service-' + apiStatus}>
          <span className="status-dot" />
          {connectionLabel}
        </div>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <div className="eyebrow"><span className="eyebrow-line" /> ACTIVE BENCHMARK</div>
          <h1>One benchmark.<br /><span>Measured honestly.</span></h1>
          <p>
            Current work trains on the paper’s abCAN training data and evaluates
            on the held-out M515 antibody–antigen mutation benchmark. Other datasets
            and model approaches remain visible
            for future evaluation.
          </p>
          <div className="hero-pills">
            <span className="pill pill-active">abCAN M515</span>
            <span className="pill">ΔΔG regression</span>
            <span className="pill">kcal/mol</span>
          </div>
        </div>

        <aside className="hero-note">
          <div className="note-icon">↗</div>
          <div>
            <span className="note-label">Published reference</span>
            <strong>RMSE 1.460 <small>kcal/mol</small></strong>
            <span className="note-sub">PCC 0.731 · M515 test set</span>
          </div>
          <a href={reference.source} target="_blank" rel="noreferrer">Read the paper ↗</a>
        </aside>
      </section>

      <section className="section-block">
        <div className="section-heading">
          <div>
            <div className="eyebrow">SCOREBOARD</div>
            <h2>Published baseline vs this project</h2>
          </div>
          <span className="section-context">
            {(activeBenchmark.name || 'abCAN M515') + ' · ' + projectStatusLabel}
          </span>
        </div>

        <div className="metrics-grid">
          <MetricCard
            label="Published abCAN RMSE"
            value={Number(reference.rmse_kcal_mol).toFixed(3)}
            unit="kcal/mol"
            tone="blue"
            caption="Reported in paper text for M515"
          />
          <MetricCard
            label="Published abCAN PCC"
            value={Number(reference.pcc).toFixed(3)}
            tone="violet"
            caption="Pearson correlation on M515"
          />
          <MetricCard
            label="This project's RMSE"
            value={current ? Number(current.rmse_kcal_mol).toFixed(3) : 'Not measured'}
            unit={current ? 'kcal/mol' : ''}
            tone="neutral"
            caption={current ? 'Held-out M515 test · n=' + current.n : 'Waiting for abCAN training data, held-out M515 data, and a trained model'}
          />
          <MetricCard
            label="This project's PCC"
            value={current ? Number(current.pcc).toFixed(3) : '—'}
            tone="neutral"
            caption={current ? 'M515 test · n=' + current.n : 'No project score has been produced yet'}
          />
        </div>

        <div className="integrity-note">
          <span className="integrity-icon">i</span>
          <p>
            Reference only: the paper’s abstract and results text report RMSE 1.460,
            while its comparison table lists 1.476. This workspace has not produced
            an M515 test score yet.
          </p>
        </div>
      </section>

      <section className="section-block scope-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">ROADMAP</div>
            <h2>Future benchmarks and approaches</h2>
          </div>
          <span className="future-caption">Kept for later evaluation</span>
        </div>

        <div className="roadmap-grid">
          <article className="roadmap-card roadmap-card-active">
            <div className="roadmap-card-top">
              <span className="roadmap-icon active-icon">A</span>
              <span className="status-tag active-tag">Active now</span>
            </div>
            <h3>abCAN M515</h3>
            <p>Published held-out antibody–antigen ΔΔG test benchmark. Training uses separate abCAN data.</p>
            <div className="roadmap-meta"><span>RMSE</span><b>1.460</b><span>PCC</span><b>0.731</b></div>
          </article>

          {futureDatasets.map((dataset) => (
            <article className="roadmap-card" key={dataset.name}>
              <div className="roadmap-card-top">
                <span className="roadmap-icon">{dataset.name.slice(0, 1)}</span>
                <span className="status-tag">{dataset.status}</span>
              </div>
              <h3>{dataset.name}</h3>
              <p>{dataset.description}</p>
              <div className="coming-line">Scores will appear here when evaluated</div>
            </article>
          ))}
        </div>

        <div className="approach-row">
          <div className="approach-title">
            <span className="eyebrow">APPROACHES</span>
            <h3>Comparison tracks</h3>
          </div>
          {futureApproaches.map((approach) => (
            <div className="approach-chip" key={approach.name}>
              <span className="approach-chip-dot" />
              <div><strong>{approach.name}</strong><small>{approach.description}</small></div>
              <span className="status-tag">Coming soon</span>
            </div>
          ))}
        </div>
      </section>

      <section className="section-block predictor-section">
        <div className="section-heading">
          <div>
            <div className="eyebrow">MODEL ACCESS</div>
            <h2>Mutation prediction</h2>
          </div>
          <span className={'status-tag ' + (predictionReady ? 'active-tag' : '')}>
            {predictionReady ? 'Model ready' : 'Prediction pipeline coming soon'}
          </span>
        </div>

        <div className="predictor-card">
          <form className="predictor-form" onSubmit={handleSubmit}>
            <label className="file-field">
              <span>Wild-type complex structure</span>
              <input
                type="file"
                accept=".pdb,.ent"
                onChange={(event) => setPdbFile(event.target.files?.[0] || null)}
              />
            </label>
            <div className="mutation-fields">
              <label>Chain<input value={mutation.chain} onChange={(event) => setMutation({ ...mutation, chain: event.target.value.toUpperCase() })} /></label>
              <label>Residue<input type="number" value={mutation.resnum} onChange={(event) => setMutation({ ...mutation, resnum: Number(event.target.value) })} /></label>
              <label>Wild type<input maxLength={1} value={mutation.wild} onChange={(event) => setMutation({ ...mutation, wild: event.target.value.toUpperCase() })} /></label>
              <label>Mutation<input maxLength={1} value={mutation.mut} onChange={(event) => setMutation({ ...mutation, mut: event.target.value.toUpperCase() })} /></label>
            </div>
            <button type="submit" disabled={!predictionReady}>
              {predictionReady ? 'Predict ΔΔG' : 'Prediction unavailable'}
            </button>
          </form>

          <div className="prediction-result" aria-live="polite">
            {result ? (
              <>
                <span className="result-label">Prediction</span>
                <strong>{Number(result.ddg_point_estimate).toFixed(2)} <small>kcal/mol</small></strong>
                <span>{result.effect}</span>
                <small>95% interval: {result.confidence_interval_95.map((value) => Number(value).toFixed(2)).join(' to ')}</small>
              </>
            ) : (
              <>
                <span className="result-label">Current status</span>
                <strong className="result-waiting">Not ready yet</strong>
                <span>A validated checkpoint and real PDB feature pipeline are required.</span>
                {errorMessage && <span className="form-error">{errorMessage}</span>}
              </>
            )}
          </div>
        </div>
      </section>

      <footer className="footer">
        <span>abCAN research benchmark</span>
        <span>Active scope: abCAN M515 · Other datasets and approaches: coming soon</span>
      </footer>
    </main>
  );
}
