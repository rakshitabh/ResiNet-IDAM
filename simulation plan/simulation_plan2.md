ResiNet-IDAM — CSV-Driven Zero-Trust Simulation
1. Core concept
The simulation is driven by a user-uploaded CSV containing events and their scheduled occurrence times.

For example, the CSV might specify that:
- Event 1 occurs 3 seconds after the simulation starts.
- Event 2 occurs 4 seconds after the simulation starts.
- Event 3 occurs 6 seconds after the simulation starts.
When the user clicks Start, the application records the simulation start time and begins a timer in HH:MM:SS format.

Each event is processed when its scheduled simulation time arrives.

The central principle: the CSV schedules the events; the existing Zero-Trust engine evaluates them.

The CSV should not directly dictate whether an event is allowed or blocked. That decision must come from the configured security logic.
2. High-level architecture
              CSV UPLOAD
                  |
                  v
        +---------------------+
        | CSV Validation      |
        | Schema + Timestamps |
        | IDs + Event Ordering|
        +---------------------+
                  |
                  v
        +---------------------+
        | Simulation Controller|
        | Start / Stop / Resume|
        | Simulation Clock    |
        +---------------------+
                  |
                  v
        +---------------------+
        | Event Scheduler     |
        | Relative Event Time |
        +---------------------+
                  |
         Scheduled time reached
                  |
                  v
        +---------------------+
        | Event Processor     |
        | User + Device + IP  |
        +---------------------+
                  |
                  v
        +---------------------+
        | Zero-Trust Engine   |
        | Trust + Risk + Policy|
        +---------------------+
                  |
                  v
        +---------------------+
        | Alert & Action      |
        | Decision Logging    |
        +---------------------+
                  |
                  v
        +---------------------+
        | Live Dashboard      |
        | Timeline + Alerts   |
        | Risk + Trust Status |
        +---------------------+
                  |
                  v
        +---------------------+
        | Downloadable Report |
        +---------------------+

This should be a separate simulation module within ResiNet-IDAM, reusing the existing Zero-Trust engine rather than implementing another one.
3. CSV dataset specification
We need a standardized CSV format so users can create their own scenarios.
Recommended columns
Column	Required?	Description	Example
event_id	Yes	Unique event identifier	EVT001
offset_seconds	Yes	Seconds after simulation start	3
event_type	Yes	Type of activity	LOGIN_ATTEMPT
user_id	Yes	Simulated user identifier	USR001
device_id	Yes	Simulated device identifier	DEV001
ip_address	Recommended	Source IP address	192.168.1.20
mac_address	Recommended	Device MAC address	AA:BB:CC:DD:EE:01
resource	Recommended	Target resource or service	PLC-01
context	Optional	Additional event context	New device detected


We can extend the schema to include other inputs required by your existing Zero-Trust engine.

Important: offset_seconds represents the event's scheduled time relative to the simulation start—not the time at which an alert is generated.
Example CSV
event_id,offset_seconds,event_type,user_id,device_id,ip_address,mac_address,resource,context
EVT001,3,LOGIN_ATTEMPT,USR001,DEV001,192.168.1.20,AA:BB:CC:DD:EE:01,SCADA-GW,Known user
EVT002,4,RESOURCE_ACCESS,USR001,DEV001,192.168.1.20,AA:BB:CC:DD:EE:01,PLC-01,Normal access pattern
EVT003,6,LOGIN_ATTEMPT,USR002,DEV009,192.168.1.99,AA:BB:CC:DD:EE:09,SCADA-GW,Unrecognized device
EVT004,10,RESOURCE_ACCESS,USR002,DEV009,192.168.1.99,AA:BB:CC:DD:EE:09,PLC-01,Unusual access pattern
EVT005,15,LOGIN_ATTEMPT,USR003,DEV003,192.168.1.30,AA:BB:CC:DD:EE:03,SCADA-GW,Repeated authentication failures

These are illustrative scenarios, not actual findings or verified policy outcomes.
CSV validation rules
Before starting the simulation, validate:
1. All required columns exist.
2. Event IDs are unique.
3. Time offsets are valid, nonnegative numbers.
4. Events are sorted by increasing offset_seconds, or the user is explicitly offered a safe sorting option.
5. Required user and device identifiers are present.
6. IP and MAC addresses are valid when supplied.
7. Event types are supported by the configured simulation.
8. Every event can be mapped to the inputs expected by the Zero-Trust engine.
If validation fails, display the row number and the specific error. Do not silently discard invalid events.

Events with identical offsets can be supported as simultaneous events, provided their processing order and interaction semantics are defined.
4. Simulation clock and event scheduling
This is one of the most important parts of the implementation.

We must distinguish three different times.
Time	Meaning
Simulation time	Elapsed time since the user clicked Start
Event time	The scheduled time at which a CSV event should execute
Wall-clock time	The actual computer time at which an event is processed


Suppose the user clicks Start at 14:30:00.

The event schedule becomes:
Offset	Scheduled event time	Action
3 seconds	14:30:03	Process EVT001
4 seconds	14:30:04	Process EVT002
6 seconds	14:30:06	Process EVT003
10 seconds	14:30:10	Process EVT004
15 seconds	14:30:15	Process EVT005


These times are illustrative.

The dashboard's timer should begin at 00:00:00 and increase as the simulation progresses.

At the same time, a secondary display can show the wall-clock start time and the calculated scheduled event time.
Recommended scheduling logic
For each event:
1. Read its offset_seconds.
2. Calculate its scheduled simulation time.
3. Wait until the simulation clock reaches that offset.
4. Dispatch the event to the event processor.
5. Invoke the existing Zero-Trust engine.
6. Record the engine's actual decision and processing timestamp.
7. Update the dashboard.
8. Continue to the next scheduled event.
The scheduler should use a monotonic clock for elapsed-time calculations rather than relying exclusively on the system clock, which can change.

The event must not execute early simply because its row appears next in the CSV.

If processing is delayed, record the scheduling delay and process overdue events in order rather than silently skipping them.
5. Start, Pause, Resume and Stop
The simulation controller should have an explicit state machine.
State	Behaviour
Ready	CSV validated; simulation has not started
Running	Simulation clock advances and events are dispatched
Paused	Simulation clock stops advancing; pending events remain scheduled
Stopped	Current execution is terminated
Completed	All scheduled events have been processed
Error	An unrecoverable simulation error occurred


Start
- Record the start time.
- Initialize elapsed simulation time to zero.
- Initialize the event queue.
- Start the dashboard timer.
- Begin processing scheduled events.
Pause
- Freeze simulation time.
- Stop dispatching new scheduled events.
- Preserve the current event position.
- Allow already-dispatched events to finish or be reconciled according to the defined processing policy.
Resume
- Continue from the previous simulation position.
- Do not reset the timer.
- Do not replay already-completed events.
- Resume pending events according to their remaining scheduled time.
Stop
- Stop scheduling new events.
- Cancel pending work that has not been dispatched.
- Reconcile any event currently being processed.
- Preserve the execution history.
- Allow the user to review the results.
Restart
A restart should create a new execution session rather than overwrite the previous run.

This allows users to repeat the same scenario and compare the results.
Adjustable speed
I recommend supporting:
- 1× — normal simulation speed.
- 2× — twice as fast.
- 5× — five times as fast.
- 10× — ten times as fast.
The speed should affect event scheduling, not the Zero-Trust risk formula.

For example, an event scheduled at simulation offset 10 seconds occurs after approximately 5 real seconds at 2× speed, assuming no processing delays.
6. Live dashboard design
I recommend a dashboard divided into five sections.
A. Simulation control panel
Display:
- Uploaded CSV filename.
- Number of validated events.
- Simulation status.
- Start, Pause, Resume, Stop and Restart controls.
- Replay speed.
- Current simulation time.
- Wall-clock start time.
The timer should be prominently visible at the top.

Example:

Simulation Time: 00:00:06

Status: RUNNING

Progress: 3 / 5 events processed
B. Event timeline
Show all scheduled events and their execution status.
Event	Scheduled time	Actual processing	Status
EVT001	00:00:03	Recorded timestamp	Processed
EVT002	00:00:04	Recorded timestamp	Processed
EVT003	00:00:06	Pending	Waiting
EVT004	00:00:10	Pending	Waiting


The actual processing timestamps should be populated when the corresponding events execute.
C. Zero-Trust monitoring
Display the assessment returned by your existing engine.

Based on the current project design, this includes:
- Anomaly severity.
- Behaviour deviation.
- Identity trust.
- Device trust.
- Asset criticality.
- Overall risk score.
- Risk category.
- Policy decision.
Only display factors that the engine actually calculates for the supplied event.

If a required input is unavailable, the dashboard should identify the missing input rather than invent a trust score.
D. Live alerts
When an event produces an alert, add it to the alert panel.

Each alert should contain:
- Event ID.
- User and device identifiers.
- Scheduled event time.
- Actual processing time.
- Detection reason.
- Risk score and category.
- Policy decision.
- Evidence and relevant metadata.
Selecting an alert should open a detailed incident view.
E. Event history
Display the events that have already executed, their outcomes, and their associated evidence.

The history should remain available after the simulation completes or stops.
7. Downloadable simulation report
When the simulation finishes or the user stops it, provide a Download Report button.

The report should reflect the execution that actually occurred.
Report contents
Section 1 — Simulation summary
- Session ID.
- Uploaded dataset filename.
- Simulation start and end times.
- Total duration.
- Configured replay speed.
- Total scheduled events.
- Successfully processed events.
- Failed or cancelled events.
Section 2 — Event execution history
- Event ID and type.
- User ID and device ID.
- Scheduled offset.
- Scheduled execution time.
- Actual processing timestamp.
- Execution status.
Section 3 — Zero-Trust assessment
- Assessment inputs.
- Individual risk contributions, where available.
- Overall risk score.
- Risk category.
- Policy decision.
Section 4 — Alerts
- Alert identifiers.
- Detection reasons.
- Event associations.
- Recorded evidence.
Section 5 — Execution integrity
- Validation errors, if any.
- Processing delays.
- Failed events.
- Cancelled events.
- Incomplete or unavailable assessments.
Recommended download formats
Format	Purpose
CSV	Event history and results for further analysis
JSON	Detailed machine-readable session and assessment data
PDF	Human-readable report for demonstrations and evaluation


For the initial implementation, I recommend CSV and JSON first, followed by PDF once the report schema is stable.

The report must distinguish processed, failed, pending and cancelled events. A stopped simulation must not be reported as though every scheduled event executed successfully.
8. Proposed implementation phases
We can implement this as a self-contained module.
Phase	Implementation	Deliverable
1	Define CSV schema and validation rules	Standardized simulation input
2	Implement simulation clock and event scheduler	Timestamp-driven execution
3	Implement execution state management	Start, Pause, Resume, Stop and Restart
4	Integrate the existing Zero-Trust engine	Actual risk assessments and policy outcomes
5	Build the live dashboard	Timer, timeline, alerts and trust visualization
6	Implement execution logging and report generation	CSV and JSON downloads, followed by PDF
7	Test timing, event ordering, error handling and repeatability	Reliable demonstration


The simulator should reuse the existing detection and Zero-Trust components. It should not implement a second risk engine or modify the core detection pipeline unnecessarily.

We can build and test it locally without Docker.
9. Important implementation decisions
Before coding, we should settle four technical details.

1. Event execution timing

The simulation clock determines when an event becomes eligible for processing. Model inference and policy evaluation may take additional time, so we should record scheduled and actual processing times separately.

2. Pause behaviour

Pausing must freeze simulation time, not merely hide the timer. Otherwise, events could execute while the dashboard appears paused.

3. Reproducibility

Every run should have its own session ID, event history and report. Replaying the same CSV with the same configuration should produce comparable results, subject to any explicitly randomized simulation inputs.

4. Safe simulation boundaries

User IDs, IP addresses and MAC addresses in the uploaded file should be treated as simulation data. The module should not attempt to access those devices or perform real authentication, blocking, scanning or isolation.

The existing Zero-Trust engine should evaluate the supplied simulated context and return its configured decision.
My final recommendation, bro
Build a CSV-driven, event-scheduled Zero-Trust simulator with a live timer, event timeline, actual risk assessments, interactive controls and downloadable execution reports.

The CSV defines when events are scheduled to occur and supplies their contextual inputs. The Zero-Trust engine determines how those events are assessed. The dashboard explains what happened and why.

That separation is what makes the demonstration both understandable and technically credible.

The next step should be to inspect the existing ResiNet-IDAM repository and map this plan to the actual files and functions before implementing anything. That will let us reuse the existing engine safely and identify the smallest set of new components required.