import { useEffect, useMemo, useState } from "react";
import { api, DatasetProfile, EventRecord, Metrics, ModelInfo, RiskLevel, Summary } from "./api";

type View = "overview" | "events" | "evaluation" | "dataset";
const levels: RiskLevel[] = ["LOW", "MEDIUM", "HIGH", "CRITICAL"];

function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

function App() {
  const [view, setView] = useState<View>("overview");
  const [summary, setSummary] = useState<Summary | null>(null);
  const [events, setEvents] = useState<EventRecord[]>([]);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [dataset, setDataset] = useState<DatasetProfile | null>(null);
  const [selected, setSelected] = useState<EventRecord | null>(null);
  const [levelFilter, setLevelFilter] = useState<RiskLevel | "ALL">("ALL");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([api.summary(), api.events(), api.modelInfo(), api.metrics(), api.datasetProfile()])
      .then(([loadedSummary, loadedEvents, loadedModel, loadedMetrics, loadedDataset]) => {
        setSummary(loadedSummary);
        setEvents(loadedEvents);
        setModel(loadedModel);
        setMetrics(loadedMetrics);
        setDataset(loadedDataset);
      })
      .catch((reason: Error) => setError(reason.message))
      .finally(() => setLoading(false));
  }, []);

  const filteredEvents = useMemo(
    () =>
      levelFilter === "ALL"
        ? events
        : events.filter((event) => event.risk_level === levelFilter),
    [events, levelFilter],
  );

  if (loading) return <div className="center-state">Loading security console…</div>;
  if (error) {
    return (
      <div className="center-state error-state">
        <strong>Dashboard unavailable</strong>
        <span>{error}</span>
        <span>Start the FastAPI service on port 8000 and refresh this page.</span>
      </div>
    );
  }

  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">R</span>
          <div><strong>ResiNet</strong><small>IDAM security console</small></div>
        </div>
        <nav>
          <button className={view === "overview" ? "active" : ""} onClick={() => setView("overview")}>Overview</button>
          <button className={view === "events" ? "active" : ""} onClick={() => setView("events")}>Event explorer</button>
          <button className={view === "evaluation" ? "active" : ""} onClick={() => setView("evaluation")}>Model evaluation</button>
          <button className={view === "dataset" ? "active" : ""} onClick={() => setView("dataset")}>Dataset profile</button>
        </nav>
        <div className="sidebar-note">
          <span className="status-dot" /> API connected
          <small>Simulated Zero-Trust context</small>
        </div>
      </aside>
      <main className="content">
        <header className="topbar">
          <div><p className="eyebrow">HAI industrial telemetry</p><h1>{view === "overview" ? "Security overview" : view === "events" ? "Event explorer" : view === "evaluation" ? "Model evaluation" : "Dataset profile"}</h1></div>
          <span className="model-pill">{model?.model_version ?? "Model unavailable"}</span>
        </header>
        {view === "overview" && <Overview summary={summary} events={events} onSelect={setSelected} onExplore={() => setView("events")} />}
        {view === "events" && <EventExplorer events={filteredEvents} level={levelFilter} setLevel={setLevelFilter} onSelect={setSelected} />}
        {view === "evaluation" && <Evaluation model={model} metrics={metrics} />}
        {view === "dataset" && <DatasetProfileView profile={dataset} />}
      </main>
      {selected && <EventModal event={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

function Overview({ summary, events, onSelect, onExplore }: { summary: Summary | null; events: EventRecord[]; onSelect: (event: EventRecord) => void; onExplore: () => void }) {
  const distribution = summary?.risk_distribution ?? {};
  return (
    <>
      <section className="hero-grid">
        <article className="hero-card"><span>Total events</span><strong>{summary?.total_events ?? 0}</strong><small>Persisted locally in SQLite</small></article>
        <article className="hero-card accent"><span>Anomalies detected</span><strong>{summary?.anomalies_detected ?? 0}</strong><small>Reconstruction error above threshold</small></article>
        <article className="hero-card"><span>High / critical</span><strong>{(distribution.HIGH ?? 0) + (distribution.CRITICAL ?? 0)}</strong><small>Prioritized for investigation</small></article>
      </section>
      <section className="panel-grid">
        <article className="panel"><div className="panel-heading"><div><p className="eyebrow">Risk posture</p><h2>Distribution</h2></div></div>
          <div className="risk-bars">{levels.map((level) => <div className="risk-row" key={level}><span className={`badge ${level.toLowerCase()}`}>{level}</span><div className="bar"><i className={level.toLowerCase()} style={{ width: `${Math.min(100, (distribution[level] ?? 0) * 10)}%` }} /></div><strong>{distribution[level] ?? 0}</strong></div>)}</div>
        </article>
        <article className="panel"><div className="panel-heading"><div><p className="eyebrow">Latest activity</p><h2>Recent events</h2></div><button className="text-button" onClick={onExplore}>View explorer →</button></div>
          {events.length === 0 ? <EmptyState text="No events have been scored yet." /> : <div className="mini-list">{events.slice(0, 5).map((event) => <button className="mini-event" key={event.event_id} onClick={() => onSelect(event)}><span className={`severity-dot ${event.risk_level.toLowerCase()}`} /><span><strong>{event.asset_id}</strong><small>{new Date(event.timestamp).toLocaleString()}</small></span><b>{event.risk_score.toFixed(1)}</b></button>)}</div>}
        </article>
      </section>
      <div className="notice"><strong>Research prototype</strong><span>Risk context is simulated. Model scores describe benchmark behavior and do not confirm malicious activity or authorize automatic industrial response.</span></div>
    </>
  );
}

function EventExplorer({ events, level, setLevel, onSelect }: { events: EventRecord[]; level: RiskLevel | "ALL"; setLevel: (level: RiskLevel | "ALL") => void; onSelect: (event: EventRecord) => void }) {
  return <section className="panel full-panel"><div className="panel-heading"><div><p className="eyebrow">Persisted security events</p><h2>Event explorer</h2></div><select value={level} onChange={(event) => setLevel(event.target.value as RiskLevel | "ALL")}><option value="ALL">All risk levels</option>{levels.map((item) => <option value={item} key={item}>{item}</option>)}</select></div>
    {events.length === 0 ? <EmptyState text="No events match this filter." /> : <div className="table-wrap"><table><thead><tr><th>Timestamp</th><th>Asset</th><th>Anomaly score</th><th>Risk score</th><th>Severity</th><th>Status</th></tr></thead><tbody>{events.map((event) => <tr key={event.event_id} onClick={() => onSelect(event)}><td>{new Date(event.timestamp).toLocaleString()}</td><td><strong>{event.asset_id}</strong></td><td>{event.anomaly_score.toFixed(4)}</td><td><strong>{event.risk_score.toFixed(1)}</strong></td><td><span className={`badge ${event.risk_level.toLowerCase()}`}>{event.risk_level}</span></td><td>{event.review_status}</td></tr>)}</tbody></table></div>}
  </section>;
}

function Evaluation({ model, metrics }: { model: ModelInfo | null; metrics: Metrics | null }) {
  if (!metrics || !model) return <EmptyState text="Evaluation artifacts are unavailable." />;
  const values = [["Precision", metrics.precision], ["Recall", metrics.recall], ["F1 score", metrics.f1], ["False-positive rate", metrics.false_positive_rate], ["ROC-AUC", metrics.roc_auc], ["PR-AUC", metrics.pr_auc]];
  return <><section className="metric-grid">{values.map(([label, value]) => <article className="metric-card" key={label as string}><span>{label}</span><strong>{formatPercent(value as number)}</strong></article>)}</section><section className="panel full-panel"><div className="panel-heading"><div><p className="eyebrow">Measured benchmark result</p><h2>Confusion matrix</h2></div><span className="muted">Threshold: {model.threshold.toFixed(4)}</span></div><div className="matrix"><div /><strong>Predicted normal</strong><strong>Predicted anomaly</strong><strong>Actual normal</strong><b>{metrics.confusion_matrix.tn.toLocaleString()}</b><b className="warn">{metrics.confusion_matrix.fp.toLocaleString()}</b><strong>Actual attack</strong><b className="warn">{metrics.confusion_matrix.fn.toLocaleString()}</b><b className="good">{metrics.confusion_matrix.tp.toLocaleString()}</b></div><div className="notice"><strong>Interpretation</strong><span>The threshold was selected from chronological normal validation errors. These metrics are HAI benchmark measurements, not a guarantee of zero-day detection.</span></div></section></>;
}

function DatasetProfileView({ profile }: { profile: DatasetProfile | null }) {
  const [search, setSearch] = useState("");
  const [group, setGroup] = useState("ALL");
  if (!profile) return <EmptyState text="Dataset profile is unavailable." />;
  const filtered = profile.features.filter((feature) => {
    const matchesSearch = `${feature.name} ${feature.process} ${feature.signal_type} ${feature.meaning}`.toLowerCase().includes(search.toLowerCase());
    return matchesSearch && (group === "ALL" || feature.process === group);
  });
  const attackPercent = (profile.test.attack_rows / profile.test.rows) * 100;
  return <><section className="hero-grid dataset-hero">
    <article className="hero-card"><span>Training observations</span><strong>{profile.training.rows.toLocaleString()}</strong><small>Normal-behavior learning interval</small></article>
    <article className="hero-card accent"><span>Test observations</span><strong>{profile.test.rows.toLocaleString()}</strong><small>Held-out evaluation interval</small></article>
    <article className="hero-card"><span>Model features</span><strong>{profile.model_input.feature_count}</strong><small>Sampled every {profile.model_input.sampling_interval_seconds} second</small></article>
  </section>
  <section className="panel-grid">
    <article className="panel"><div className="panel-heading"><div><p className="eyebrow">Label composition</p><h2>Evaluation population</h2></div></div><div className="composition"><div className="composition-bar"><i style={{ width: `${100 - attackPercent}%` }} /><i style={{ width: `${attackPercent}%` }} /></div><div className="legend"><span><i className="legend-normal" /> Normal {profile.test.normal_rows.toLocaleString()} ({(100 - attackPercent).toFixed(1)}%)</span><span><i className="legend-attack" /> Attack {profile.test.attack_rows.toLocaleString()} ({attackPercent.toFixed(1)}%)</span></div><p className="small-copy">{profile.label_usage}</p></div></article>
    <article className="panel"><div className="panel-heading"><div><p className="eyebrow">Signal families</p><h2>Feature groups</h2></div></div><div className="group-list">{profile.feature_groups.map((item) => <div key={item.name}><span>{item.name}</span><strong>{item.count}</strong></div>)}</div></article>
  </section>
  <section className="panel full-panel"><div className="panel-heading"><div><p className="eyebrow">What the pipeline measures</p><h2>Derived measurements</h2></div></div><div className="measurement-grid">{profile.measurements.map((item) => <article key={item.name}><span>{item.unit}</span><h3>{item.name}</h3><p>{item.meaning}</p></article>)}</div></section>
  <section className="panel full-panel"><div className="panel-heading"><div><p className="eyebrow">66 ordered model inputs</p><h2>Feature dictionary</h2></div><span className="muted">{filtered.length} shown</span></div><div className="filters"><input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search feature, process, or meaning" /><select value={group} onChange={(event) => setGroup(event.target.value)}><option value="ALL">All process groups</option>{profile.feature_groups.map((item) => <option value={item.name} key={item.name}>{item.name}</option>)}</select></div><div className="table-wrap"><table><thead><tr><th>Feature</th><th>Process</th><th>Signal type</th><th>Meaning</th><th>Model use</th></tr></thead><tbody>{filtered.map((feature) => <tr key={feature.name}><td><strong>{feature.name}</strong></td><td>{feature.process}</td><td>{feature.signal_type}</td><td>{feature.meaning}</td><td>{feature.model_usage}</td></tr>)}</tbody></table></div></section>
  </>;
}

function EventModal({ event, onClose }: { event: EventRecord; onClose: () => void }) {
  return <div className="modal-backdrop" onClick={onClose}><article className="modal" onClick={(click) => click.stopPropagation()}><button className="close-button" onClick={onClose}>×</button><p className="eyebrow">Event detail</p><h2>{event.asset_id}</h2><span className={`badge ${event.risk_level.toLowerCase()}`}>{event.risk_level} · {event.risk_score.toFixed(1)}/100</span><dl><dt>Timestamp</dt><dd>{new Date(event.timestamp).toLocaleString()}</dd><dt>Anomaly score / threshold</dt><dd>{event.anomaly_score.toFixed(5)} / {event.threshold.toFixed(5)}</dd><dt>Identity / device</dt><dd>{event.identity_id} / {event.device_id}</dd><dt>Review status</dt><dd>{event.review_status}</dd></dl><h3>Risk components</h3><div className="component-list">{Object.entries(event.risk_components).map(([name, value]) => <div key={name}><span>{name.replaceAll("_", " ")}</span><strong>{value.toFixed(2)}</strong></div>)}</div><small className="simulated">Identity, device, and asset context is simulated.</small></article></div>;
}

function EmptyState({ text }: { text: string }) { return <div className="empty-state">{text}</div>; }

export default App;
