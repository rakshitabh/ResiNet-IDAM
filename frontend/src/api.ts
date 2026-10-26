export type RiskLevel = "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";

export interface EventRecord {
  event_id: string;
  timestamp: string;
  asset_id: string;
  anomaly_score: number;
  threshold: number;
  anomaly_decision: boolean;
  risk_score: number;
  risk_level: RiskLevel;
  review_status: string;
  identity_id: string;
  device_id: string;
  context: Record<string, number>;
  risk_components: Record<string, number>;
}

export interface Summary {
  total_events: number;
  anomalies_detected: number;
  risk_distribution: Partial<Record<RiskLevel, number>>;
}

export interface ModelInfo {
  model_version: string;
  threshold: number;
  threshold_selection: string;
  feature_count: number;
  feature_columns: string[];
}

export interface Metrics {
  precision: number;
  recall: number;
  f1: number;
  false_positive_rate: number;
  roc_auc: number;
  pr_auc: number;
  confusion_matrix: { tn: number; fp: number; fn: number; tp: number };
}

export interface DatasetFeature {
  name: string;
  process: string;
  signal_type: string;
  meaning: string;
  model_usage: string;
  train_mean: number;
  train_std: number;
  test_mean: number;
  test_std: number;
  mean_difference: number;
}

export interface DatasetProfile {
  dataset: string;
  source: string;
  training: { file: string; rows: number; columns: number };
  test: { file: string; rows: number; columns: number; normal_rows: number; attack_rows: number };
  model_input: { feature_count: number; excluded_columns: string[]; sampling_interval_seconds: number };
  measurements: { name: string; unit: string; meaning: string }[];
  label_usage: string;
  feature_groups: { name: string; count: number }[];
  features: DatasetFeature[];
}

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000/api";

async function get<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`);
  if (!response.ok) {
    throw new Error(`API request failed (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  summary: () => get<Summary>("/summary"),
  events: () => get<EventRecord[]>("/events?limit=100"),
  modelInfo: () => get<ModelInfo>("/model/info"),
  metrics: () => get<Metrics>("/model/metrics"),
  datasetProfile: () => get<DatasetProfile>("/dataset/profile"),
  event: (id: string) => get<EventRecord>(`/events/${encodeURIComponent(id)}`),
};
