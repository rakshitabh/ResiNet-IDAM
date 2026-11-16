ResiNet-IDAM: Simulation Module — Implementation Plan
1. Objective
Build an interactive, timestamp-driven simulation that demonstrates how an industrial anomaly progresses through detection, Zero-Trust risk assessment, and simulated security decision-making.

The simulation must support two types of users:
- Beginner: Understands what happened, why it matters, and what the system decided.
- Technical user: Can inspect timestamps, sensor values, anomaly scores, model outputs, trust factors, and policy decisions.
Core principle
One underlying event, multiple levels of explanation.

We should not build separate simulation logic for beginners and technical users. Both views should display evidence from the same underlying event.
2. Proposed architecture
The simulation should integrate with your existing ResiNet-IDAM components.
                  USER
                   |
                   v
        +----------------------+
        | Simulation Dashboard |
        | Streamlit UI         |
        +----------------------+
                   |
          Start / Pause / Resume
          Speed / Scenario / Seek
                   |
                   v
        +----------------------+
        | Simulation Controller|
        | Replay State & Clock |
        +----------------------+
                   |
                   v
        +----------------------+
        | Dataset Replay Engine|
        | Chronological Records|
        +----------------------+
                   |
                   v
        +----------------------+
        | Existing Detection   |
        | Pipeline             |
        | IT / OT Models       |
        +----------------------+
                   |
                   v
        +----------------------+
        | Existing Zero-Trust  |
        | Risk Engine          |
        +----------------------+
                   |
                   v
        +----------------------+
        | Incident & Event      |
        | Management            |
        +----------------------+
                   |
                   v
        +----------------------+
        | Existing SQLite      |
        | Event Store          |
        +----------------------+
                   |
                   v
        +----------------------+
        | Visualization Layer  |
        | Timeline / Process   |
        | Evidence / Decisions |
        +----------------------+

Important: This is the proposed logical architecture, not a claim that these components do not already exist. During the initial audit, we'll identify which components can be reused.

The simulation must not modify the existing model's detection logic merely to make the demonstration visually interesting.
3. Seven implementation phases
Phase 1 — Audit the existing system
Goal: Identify the components already available for simulation.

Inspect the following:
Component	What we need to verify
Dataset loader	Can it read prepared IT and OT datasets?
Detection pipeline	Can it process observations individually or in chronological batches?
Zero-Trust engine	What inputs does it accept, and what does it return?
Event store	What information is persisted for each alert?
Dashboard	Which charts, tables, and alert views already exist?
Dataset labels	How are ground-truth labels separated from model inputs?
Tests	Which behaviours already have automated coverage?


Deliverable: A gap analysis identifying existing functionality, required extensions, and genuinely new components.

Acceptance criterion: We can trace one observation through the existing pipeline and identify the data required to display its result.

Do not modify the production code or stored data during this audit.
Phase 2 — Design the simulation data model
Goal: Establish a consistent representation of observations, events, and incidents.

We need four logical entities.
A. Simulation session
Represents one execution of a dataset replay.

Suggested fields:
session_id
dataset_id
domain
simulation_status
replay_speed
start_timestamp
current_timestamp
configuration

B. Observation
Represents an individual record from the dataset.
observation_id
session_id
observation_timestamp
feature_values
ground_truth_label

The ground-truth label must be kept separate from the features passed to the detector.
C. Detection event
Represents the detector's actual output.
event_id
session_id
observation_timestamp
detection_timestamp
model_name
model_version
anomaly_score
threshold
detection_result

D. Incident
Represents a security event and its investigation context.
incident_id
session_id
first_detection_timestamp
last_detection_timestamp
severity
risk_score
risk_level
trust_assessment
policy_decision
evidence_reference
incident_status

These are proposed logical schemas. We should adapt them to the existing SQLite schema instead of creating duplicate tables unnecessarily.

Deliverable: A documented data model and event contract.

Acceptance criterion: Every displayed incident can be traced to its simulation session, detector output, risk assessment, and underlying observations.
Phase 3 — Implement the timestamp-driven replay engine
Goal: Replay the dataset in chronological order.

The user should be able to:
- Start the simulation.
- Pause and resume it.
- Change the replay speed.
- View the current timestamp.
- Inspect replay progress.
- Select an available replay interval.
- Restart a session reproducibly.
How replay speed should work
Suppose the recorded observations occur at one-second intervals.
Replay mode	Simulation behaviour
1×	One simulated second per real second
5×	Five simulated seconds per real second
20×	Twenty simulated seconds per real second
Step mode	Process the next observation or configured time window manually


These are proposed controls.

The simulation clock must be based on the dataset timestamps, not the time taken by the computer to process the model.

If inference takes longer than the replay interval, the engine should process observations in order without silently dropping records.

For seeking and restarting, use a deterministic replay position and configuration. When the user seeks directly to an incident, the system should reconstruct the necessary context without pretending it processed every preceding observation in real time.

Deliverable: A replay controller connected to the existing dataset loader.

Acceptance criterion: The same dataset and model configuration produce reproducible outputs, regardless of the visualization speed.
Phase 4 — Connect replay to detection and Zero Trust
Goal: Turn actual detector outputs into visible security events.

This is the core of the simulation.

The processing sequence should be:
1. Read the next observation.
2. Validate its timestamp and feature schema.
3. Pass the permitted features through the existing detection pipeline.
4. Record the model output and detection decision.
5. When an alert is raised, obtain the corresponding Zero-Trust assessment.
6. Record the resulting risk score and policy decision.
7. Create or update the associated incident.
8. Notify the visualization layer.
Example
Suppose an OT observation is processed at a dataset timestamp of 10:00:04.

The detector returns an anomaly score that exceeds its configured threshold.

The simulation then displays:
Field	Display
Observation time	10:00:04
Detection result	Anomaly detected
Anomaly score	Actual model output
Detection threshold	Actual configured threshold
Zero-Trust assessment	Actual risk-engine result
Policy decision	Actual configured policy outcome


The timestamp and numerical values above are illustrative. The implementation must use actual results.
Avoid duplicate incidents
A sequence of consecutive anomalous observations should not automatically create a separate incident card for every observation.

Implement configurable incident grouping:
- Open: First qualifying detection.
- Update: Additional related detections.
- Close: The configured resolution condition is met.
Preserve the individual detection records even when they are grouped into one incident.
Preserve the safety boundary
The simulation must not send commands to real industrial controllers.

Its policy outcomes should be explicitly labelled as simulated decisions.

Deliverable: An event-driven incident pipeline using the existing detector and risk engine.

Acceptance criterion: The simulation displays actual model and risk-engine outputs, with no fabricated detection results.
4. Phase 5 — Build the visualization
This phase is where your original idea becomes valuable.

I recommend four connected visualization components.
A. Industrial-process view
Show a simplified representation of the monitored industrial process.

For example:











The visualization can contain:
- Process components.
- Sensor indicators.
- Normal and anomalous states.
- An incident highlight.
- A link to the corresponding sensor observations.
Initially, this should be a 2D process diagram, not a complex 3D environment.

However, there is an important limitation: we must verify whether the available HAI features can be mapped meaningfully to specific process components. If that mapping is not supported, we should display an abstract process view with clearly identified sensor groups rather than inventing physical relationships.
B. Incident timeline
Display the sequence of events.
10:00:01  Normal observations
    |
10:00:02  Normal observations
    |
10:00:03  Behaviour begins to change
    |
10:00:04  Anomaly detected
    |
10:00:04  Zero-Trust assessment performed
    |
10:00:05  Additional evidence collected
    |
            Incident investigation

This is an illustrative sequence, not a claim about an actual HAI incident.

The timeline should distinguish:
- Observation time.
- Detection time.
- Risk-assessment time.
- Policy-decision time.
A dataset may contain an anomalous observation before the detector recognizes it. The visualization must preserve that distinction.
C. Sensor behaviour
Display time-series graphs for relevant features.

For each selected sensor, show:
- Observed values.
- The relevant historical window.
- The anomaly detection timestamp.
- The detector threshold or score, where applicable.
- A comparison with an appropriate baseline, if one is available.
Do not show a universal anomaly threshold on every sensor. An autoencoder's reconstruction-error threshold is not necessarily a threshold for the raw sensor value.
D. Zero-Trust assessment
Display the assessment using the existing five-factor risk engine.

Your existing risk model uses these weights:
Factor	Weight
Anomaly severity	40%
Behaviour deviation	20%
Identity untrust	20%
Device untrust	10%
Asset criticality	10%
Total	100%


These weights come from the existing project context; we should verify the implementation before displaying them as definitive.

The interface should show each factor's contribution, the overall risk score, the risk category, and the resulting simulated policy decision.

Deliverable: A unified visualization connecting process activity, detection evidence, incident chronology, and Zero-Trust assessment.

Acceptance criterion: A beginner can understand the incident, and a technical user can inspect the evidence supporting it.
Phase 6 — Add guided investigation and explanation
Goal: Help users who do not know the underlying industrial system understand the incident.

The dashboard should offer three views.
View	Primary audience	Content
Beginner	Evaluators and new users	What happened, why it matters, risk, and outcome
Analyst	Cybersecurity practitioners	Sensor graphs, anomaly scores, risk factors, incident timeline
Researcher	Technical reviewers	Model outputs, feature values, thresholds, event records, and ground-truth comparison


Beginner example
When an incident occurs, the dashboard could display:

What happened?

The anomaly detector identified an observation that differed from the behaviour learned by the model.

Why was it flagged?

The model's reconstruction error exceeded the configured threshold.

What did the security engine do?

The Zero-Trust engine evaluated the configured risk factors and produced a risk assessment.

What was the outcome?

The simulated policy engine selected the configured response.

The displayed explanation must be generated from the actual event data. It should not assert that an attacker caused a change unless the available evidence supports that conclusion.
Incident replay
Users should be able to select an incident and inspect:
1. The observations preceding detection.
2. The observation that triggered the alert.
3. The detector's output.
4. The Zero-Trust assessment.
5. The subsequent observations and incident outcome.
The preceding observations provide context; they must not be presented as proof that an attack began at a particular instant unless that conclusion is supported by the data.

Deliverable: An explainable incident-investigation workflow.

Acceptance criterion: Users can move from a plain-language explanation to the underlying technical evidence without switching to a separate system.
Phase 7 — Testing and validation
Goal: Verify that the simulation is accurate, reproducible, and understandable.
Test	Expected result
Normal replay	Observations are processed in chronological order.
Detected anomaly	The actual detector output triggers the event workflow.
No detected anomaly	No fabricated detection is displayed.
Replay pause	The simulation clock stops advancing.
Replay resume	Processing continues from the correct position.
Replay restart	Results are reproducible under the same configuration.
Consecutive anomalies	Related detections are grouped without losing individual evidence.
Risk assessment	The displayed factors and score match the risk engine.
Ground-truth comparison	Labels are used for evaluation, not secretly passed into detection.
Incident investigation	The timeline links to the correct observations and model outputs.
Performance	Replay speed, processing latency, and dropped or delayed observations are measured.
Safety	No real industrial-control action is executed.


We should also validate the beginner experience with a few representative scenarios. A dashboard is not genuinely explainable merely because it contains many charts.

Deliverable: A reproducible local demonstration and a test report.
5. Suggested implementation order
The seven phases above describe the complete work. I recommend implementing them through four practical milestones.
Milestone	Work included	Completion condition
M1 — Foundation	Phase 1 and Phase 2	Existing components audited; data model finalized.
M2 — Replay and detection	Phase 3 and Phase 4	Dataset replay generates real detection events and risk assessments.
M3 — Visualization	Phase 5 and Phase 6	Users can view, understand, and investigate incidents.
M4 — Validation	Phase 7	Automated tests pass and the demonstration is reproducible.


Technology choices
For the first implementation:
- Python: Dataset replay and coordination with the existing detection pipeline.
- Existing detection models: No unnecessary replacement.
- Existing Zero-Trust engine: No duplicated risk calculations.
- SQLite: Reuse the existing event store where appropriate.
- Streamlit: Extend the existing operations dashboard.
- Existing plotting libraries: Reuse them if they meet the visualization requirements.
We should not introduce Docker, a second backend, a separate database, or a new model until a concrete requirement justifies it.