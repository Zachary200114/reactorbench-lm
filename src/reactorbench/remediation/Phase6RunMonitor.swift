import AppKit
import Foundation

private let applicationTitle = "ReactorBench-LM — Phase 6 Run Monitor"
private let expectedOfficialRunName = "phase6-remediation-v0.4.0-targeted-05"
private let expectedDiagnosticRunName = "phase6-remediation-v0.4.0-targeted-05-diagnostic-01"
private let maximumHelperOutputBytes = 32 * 1024
private let maximumActivityEntries = 100
private let maximumActivityCharacters = 32 * 1024
private let terminalFailureStates: Set<String> = ["Blocked", "Failed"]
private let terminalFailureAlarmDurationSeconds = 45.0
private let terminalFailureFallbackBeepCount = 23
private let terminalFailureFallbackBeepSpacingSeconds = 2.0
private let terminalFailureSoundPath = "/System/Library/Sounds/Sosumi.aiff"

private enum HelperAction: String {
    case snapshot = "--snapshot-json"
    case readiness = "--readiness-check"
    case start = "--start-detached"
    case stop = "--request-stop"
    case resume = "--resume-detached"
    case finder = "--open-finder"
    case diagnosticSnapshot = "--diagnostic-snapshot-json"
    case diagnosticReadiness = "--diagnostic-readiness-check"
    case diagnosticStart = "--diagnostic-start-detached"
    case diagnosticStop = "--diagnostic-request-stop"
    case diagnosticResume = "--diagnostic-resume-detached"
    case diagnosticFinder = "--diagnostic-open-finder"
}

private enum MonitorTarget {
    case official
    case diagnostic

    var expectedRunName: String {
        self == .official ? expectedOfficialRunName : expectedDiagnosticRunName
    }
}

private struct PolicyPayload: Decodable {
    let copyStatus: Bool
    let dryRun: Bool
    let openFinder: Bool
    let refresh: Bool
    let resume: Bool
    let start: Bool
    let stop: Bool
}

private struct StatusPayload: Decodable {
    let completedUnits: Int?
    let currentStage: String
    let elapsedSeconds: Double?
    let etaSeconds: Double?
    let eventSequence: Int?
    let interruptions: Int
    let latestCheckpoint: String?
    let latestEvent: String
    let latestMessage: String
    let latestUpdateUtc: String?
    let metricName: String?
    let metricValue: Double?
    let nextStage: String
    let overallPercent: Double
    let pipelineStatus: String
    let policy: PolicyPayload
    let runExists: Bool
    let runName: String
    let safeError: String?
    let sourceCommit: String
    let stageIndex: Int?
    let stageTotal: Int?
    let state: String
    let stopRequested: Bool
    let summary: String
    let totalUnits: Int?
    let verified: Bool
    let versionName: String
    let versionPercent: Double
    let versionStageIndex: Int
    let versionStageTotal: Int
    let workPercent: Double
}

private struct OperationPayload: Decodable {
    let kind: String
    let message: String
    let ok: Bool
    let pid: Int?
    let returncode: Int
}

private enum MonitorDecodeError: Error {
    case invalidShape
    case oversized
}

private let statusKeys: Set<String> = [
    "completed_units", "current_stage", "elapsed_seconds", "eta_seconds",
    "event_sequence", "interruptions", "latest_checkpoint", "latest_event",
    "latest_message", "latest_update_utc", "metric_name", "metric_value",
    "next_stage", "overall_percent", "pipeline_status", "policy", "run_exists",
    "run_name", "safe_error", "source_commit", "stage_index", "stage_total",
    "state", "stop_requested", "summary", "total_units", "verified", "version_name",
    "version_percent", "version_stage_index", "version_stage_total", "work_percent",
]

private let policyKeys: Set<String> = [
    "copy_status", "dry_run", "open_finder", "refresh", "resume", "start", "stop",
]

private let operationKeys: Set<String> = ["kind", "message", "ok", "pid", "returncode"]

private func checkedDictionary(_ data: Data, expectedKeys: Set<String>) throws -> [String: Any] {
    guard data.count > 0, data.count <= maximumHelperOutputBytes else {
        throw MonitorDecodeError.oversized
    }
    let object = try JSONSerialization.jsonObject(with: data)
    guard let dictionary = object as? [String: Any], Set(dictionary.keys) == expectedKeys else {
        throw MonitorDecodeError.invalidShape
    }
    return dictionary
}

private func decodeStatus(_ data: Data) throws -> StatusPayload {
    let dictionary = try checkedDictionary(data, expectedKeys: statusKeys)
    guard
        let policy = dictionary["policy"] as? [String: Any],
        Set(policy.keys) == policyKeys
    else {
        throw MonitorDecodeError.invalidShape
    }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let payload = try decoder.decode(StatusPayload.self, from: data)
    guard
        [expectedOfficialRunName, expectedDiagnosticRunName].contains(payload.runName),
        payload.overallPercent.isFinite,
        payload.versionPercent.isFinite,
        payload.workPercent.isFinite,
        (0 ... 100).contains(payload.overallPercent),
        (0 ... 100).contains(payload.versionPercent),
        ["Setup", "v0.2", "v0.3", "v0.4", "Finalization", "Complete"].contains(
            payload.versionName
        ),
        payload.versionStageTotal > 0,
        (0 ... payload.versionStageTotal).contains(payload.versionStageIndex),
        (0 ... 100).contains(payload.workPercent)
    else {
        throw MonitorDecodeError.invalidShape
    }
    return payload
}

private func decodeOperation(_ data: Data) throws -> OperationPayload {
    _ = try checkedDictionary(data, expectedKeys: operationKeys)
    return try JSONDecoder().decode(OperationPayload.self, from: data)
}

private final class MonitorWindowController: NSWindowController, NSWindowDelegate {
    private let projectRoot: URL
    private let helperSource: URL
    private let controllerWrapper: URL

    private let stateLabel = NSTextField(labelWithString: "Checking…")
    private let stageLabel = NSTextField(labelWithString: "Stage unavailable")
    private let overallLabel = NSTextField(
        labelWithString: "ENTIRE RERUN — all versions and all 16 stages: 0.0%"
    )
    private let versionLabel = NSTextField(labelWithString: "Current version — Setup: 0.0%")
    private let workLabel = NSTextField(labelWithString: "Current work: unavailable")
    private let overallProgress = NSProgressIndicator()
    private let versionProgress = NSProgressIndicator()
    private let workProgress = NSProgressIndicator()
    private var detailLabels: [String: NSTextField] = [:]
    private let activityView = NSTextView()
    private let targetControl = NSSegmentedControl(
        labels: ["Official fail-fast", "Diagnostic full sweep"],
        trackingMode: .selectOne,
        target: nil,
        action: nil
    )

    private let readinessButton = NSButton(title: "Readiness check", target: nil, action: nil)
    private let startButton = NSButton(title: "Start new rerun", target: nil, action: nil)
    private let refreshButton = NSButton(title: "Refresh status", target: nil, action: nil)
    private let stopButton = NSButton(title: "Request safe stop", target: nil, action: nil)
    private let resumeButton = NSButton(title: "Resume stopped rerun", target: nil, action: nil)
    private let finderButton = NSButton(title: "Open run folder", target: nil, action: nil)
    private let copyButton = NSButton(title: "Copy status", target: nil, action: nil)
    private let silenceAlarmButton = NSButton(title: "Stop alarm", target: nil, action: nil)
    private let closeButton = NSButton(title: "Close", target: nil, action: nil)

    private var latestStatus: StatusPayload?
    private var activity: [String] = []
    private var operationInProgress = false
    private var statusInProgress = false
    private var detachedLaunchPending = false
    private var refreshTimer: Timer?
    private var lastStatusSignature: String?
    private var terminalFailureAlertIssued = false
    private var terminalFailureAlarm: NSSound?
    private var terminalFailureAlarmActive = false
    private var terminalFailureAlarmWorkItems: [DispatchWorkItem] = []
    private var selectedTarget: MonitorTarget = .official

    static func make() throws -> MonitorWindowController {
        var root = URL(fileURLWithPath: #filePath)
        for _ in 0 ..< 4 {
            root.deleteLastPathComponent()
        }
        let projectRoot = root.standardizedFileURL
        let helperSource = projectRoot.appendingPathComponent(
            "src/reactorbench/remediation/local_monitor.py"
        )
        let controllerWrapper = projectRoot.appendingPathComponent(
            "scripts/phase6_monitor_controller.sh"
        )
        guard
            FileManager.default.fileExists(atPath: projectRoot.appendingPathComponent("pyproject.toml").path),
            FileManager.default.isReadableFile(atPath: helperSource.path),
            FileManager.default.isExecutableFile(atPath: controllerWrapper.path)
        else {
            throw MonitorDecodeError.invalidShape
        }
        return MonitorWindowController(
            projectRoot: projectRoot,
            helperSource: helperSource,
            controllerWrapper: controllerWrapper
        )
    }

    private init(projectRoot: URL, helperSource: URL, controllerWrapper: URL) {
        self.projectRoot = projectRoot
        self.helperSource = helperSource
        self.controllerWrapper = controllerWrapper

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 980, height: 800),
            styleMask: [.titled, .closable, .miniaturizable, .resizable],
            backing: .buffered,
            defer: false
        )
        window.title = applicationTitle
        window.minSize = NSSize(width: 820, height: 680)
        window.center()
        super.init(window: window)
        window.delegate = self
        buildInterface(in: window)
        targetControl.selectedSegment = 0
        startButton.title = "Start official fail-fast run"
        appendActivity("Monitor opened. No model training was started by opening this window.")
        applyButtonState()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        nil
    }

    func begin() {
        showWindow(nil)
        window?.makeKeyAndOrderFront(nil)
        NSApplication.shared.activate(ignoringOtherApps: true)
        refreshStatus()
    }

    private func buildInterface(in window: NSWindow) {
        guard let content = window.contentView else { return }
        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 10
        stack.translatesAutoresizingMaskIntoConstraints = false
        content.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: content.leadingAnchor, constant: 18),
            stack.trailingAnchor.constraint(equalTo: content.trailingAnchor, constant: -18),
            stack.topAnchor.constraint(equalTo: content.topAnchor, constant: 16),
            stack.bottomAnchor.constraint(lessThanOrEqualTo: content.bottomAnchor, constant: -16),
        ])

        let title = NSTextField(labelWithString: applicationTitle)
        title.font = NSFont.systemFont(ofSize: 21, weight: .bold)
        stack.addArrangedSubview(title)
        targetControl.target = self
        targetControl.action = #selector(targetChanged)
        stack.addArrangedSubview(targetControl)

        let localOnly = NSTextField(
            labelWithString: "LOCAL DEVELOPMENT UTILITY — not Phase 7 and not an operational system"
        )
        localOnly.textColor = NSColor.systemOrange
        localOnly.font = NSFont.systemFont(ofSize: 12, weight: .medium)
        stack.addArrangedSubview(localOnly)

        stateLabel.font = NSFont.systemFont(ofSize: 19, weight: .bold)
        stack.addArrangedSubview(stateLabel)
        stageLabel.font = NSFont.monospacedSystemFont(ofSize: 13, weight: .regular)
        stack.addArrangedSubview(stageLabel)

        let progressContent = NSStackView()
        progressContent.orientation = .vertical
        progressContent.alignment = .leading
        progressContent.spacing = 6
        overallLabel.font = NSFont.systemFont(ofSize: 14, weight: .bold)
        progressContent.addArrangedSubview(overallLabel)
        configureProgress(overallProgress, height: 18)
        progressContent.addArrangedSubview(overallProgress)
        versionLabel.font = NSFont.systemFont(ofSize: 12, weight: .medium)
        progressContent.addArrangedSubview(versionLabel)
        configureProgress(versionProgress, height: 12)
        progressContent.addArrangedSubview(versionProgress)
        progressContent.addArrangedSubview(workLabel)
        configureProgress(workProgress, height: 12)
        progressContent.addArrangedSubview(workProgress)
        let progressBox = makeBox(title: "Entire run, current version, and current task", content: progressContent)
        stack.addArrangedSubview(progressBox)
        progressBox.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
        overallProgress.widthAnchor.constraint(equalTo: progressContent.widthAnchor).isActive = true
        versionProgress.widthAnchor.constraint(equalTo: progressContent.widthAnchor).isActive = true
        workProgress.widthAnchor.constraint(equalTo: progressContent.widthAnchor).isActive = true

        let detailRows = [
            ("Message", "message", "Latest event", "event"),
            ("Elapsed active", "elapsed", "Estimated remaining", "eta"),
            ("Latest metric", "metric", "Latest checkpoint", "checkpoint"),
            ("Interruptions", "interruptions", "Stop request", "stop"),
            ("Source commit", "commit", "Run name", "run"),
            ("Latest verified update", "updated", "Pipeline state", "pipeline"),
        ]
        let gridRows: [[NSView]] = detailRows.map { leftTitle, leftKey, rightTitle, rightKey in
            let leftValue = detailValue(for: leftKey)
            let rightValue = detailValue(for: rightKey)
            return [
                detailTitle(leftTitle), leftValue,
                detailTitle(rightTitle), rightValue,
            ]
        }
        let detailGrid = NSGridView(views: gridRows)
        detailGrid.rowSpacing = 5
        detailGrid.columnSpacing = 9
        detailGrid.column(at: 0).width = 128
        detailGrid.column(at: 1).width = 280
        detailGrid.column(at: 2).width = 140
        detailGrid.column(at: 3).width = 280
        let detailBox = makeBox(title: "Verified status", content: detailGrid)
        stack.addArrangedSubview(detailBox)
        detailBox.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true

        configureButtons()
        let buttonRows: [[NSView]] = [
            [readinessButton, startButton, refreshButton],
            [stopButton, resumeButton, silenceAlarmButton],
            [finderButton, copyButton, closeButton],
        ]
        let buttonGrid = NSGridView(views: buttonRows)
        buttonGrid.rowSpacing = 7
        buttonGrid.columnSpacing = 7
        for column in 0 ..< 3 {
            buttonGrid.column(at: column).width = 280
        }
        let controlsBox = makeBox(title: "Owner controls", content: buttonGrid)
        stack.addArrangedSubview(controlsBox)
        controlsBox.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true

        activityView.isEditable = false
        activityView.isSelectable = true
        activityView.font = NSFont.monospacedSystemFont(ofSize: 11, weight: .regular)
        activityView.textContainerInset = NSSize(width: 6, height: 6)
        let activityScroll = NSScrollView()
        activityScroll.hasVerticalScroller = true
        activityScroll.borderType = .bezelBorder
        activityScroll.documentView = activityView
        activityScroll.heightAnchor.constraint(equalToConstant: 120).isActive = true
        let activityBox = makeBox(title: "Activity", content: activityScroll)
        stack.addArrangedSubview(activityBox)
        activityBox.widthAnchor.constraint(equalTo: stack.widthAnchor).isActive = true
    }

    private func configureProgress(_ progress: NSProgressIndicator, height: CGFloat) {
        progress.style = .bar
        progress.isIndeterminate = false
        progress.minValue = 0
        progress.maxValue = 100
        progress.doubleValue = 0
        progress.heightAnchor.constraint(equalToConstant: height).isActive = true
    }

    private func makeBox(title: String, content: NSView) -> NSBox {
        let box = NSBox()
        box.title = title
        box.boxType = .primary
        box.contentViewMargins = NSSize(width: 12, height: 10)
        box.contentView = content
        return box
    }

    private func detailTitle(_ title: String) -> NSTextField {
        let label = NSTextField(labelWithString: "\(title):")
        label.font = NSFont.systemFont(ofSize: 12, weight: .medium)
        return label
    }

    private func detailValue(for key: String) -> NSTextField {
        let label = NSTextField(wrappingLabelWithString: "Unavailable")
        label.font = NSFont.systemFont(ofSize: 12)
        label.maximumNumberOfLines = 2
        detailLabels[key] = label
        return label
    }

    private func configureButtons() {
        let specifications: [(NSButton, Selector)] = [
            (readinessButton, #selector(readinessPressed)),
            (startButton, #selector(startPressed)),
            (refreshButton, #selector(refreshPressed)),
            (stopButton, #selector(stopPressed)),
            (resumeButton, #selector(resumePressed)),
            (finderButton, #selector(finderPressed)),
            (copyButton, #selector(copyPressed)),
            (silenceAlarmButton, #selector(silenceAlarmPressed)),
            (closeButton, #selector(closePressed)),
        ]
        for (button, action) in specifications {
            button.target = self
            button.action = action
            button.bezelStyle = .rounded
        }
    }

    @objc private func readinessPressed() {
        runOperation(selectedTarget == .official ? .readiness : .diagnosticReadiness)
    }

    @objc private func startPressed() {
        let diagnostic = selectedTarget == .diagnostic
        guard confirm(
            title: diagnostic
                ? "Start diagnostic full sweep?"
                : "Start official fail-fast Phase 6 run?",
            message: diagnostic
                ? "This diagnostic run continues through allowlisted scientific gate misses " +
                    "to collect all development failures. Code, integrity, resource, and safety " +
                    "failures still stop it. It cannot certify the model or unlock final evaluation."
                : "This begins the official fail-fast local model-training workflow. A scientific " +
                    "gate miss stops later stages. Continue?",
            confirmTitle: diagnostic ? "Start diagnostic sweep" : "Start official run"
        ) else { return }
        detachedLaunchPending = true
        runOperation(diagnostic ? .diagnosticStart : .start)
    }

    @objc private func refreshPressed() {
        refreshStatus()
    }

    @objc private func stopPressed() {
        guard confirm(
            title: "Request a safe stop?",
            message: "The active stage will stop at its next safe boundary. " +
                "This is not an immediate kill.",
            confirmTitle: "Request safe stop"
        ) else { return }
        runOperation(selectedTarget == .official ? .stop : .diagnosticStop)
    }

    @objc private func resumePressed() {
        guard confirm(
            title: "Resume the stopped rerun?",
            message: "The existing checksum-bound rerun will continue from its latest " +
                "verified boundary.",
            confirmTitle: "Resume"
        ) else { return }
        detachedLaunchPending = true
        runOperation(selectedTarget == .official ? .resume : .diagnosticResume)
    }

    @objc private func finderPressed() {
        runOperation(selectedTarget == .official ? .finder : .diagnosticFinder)
    }

    @objc private func targetChanged() {
        silenceTerminalFailureAlarm(recordActivity: false)
        terminalFailureAlertIssued = false
        selectedTarget = targetControl.selectedSegment == 1 ? .diagnostic : .official
        latestStatus = nil
        lastStatusSignature = nil
        detachedLaunchPending = false
        startButton.title = selectedTarget == .diagnostic
            ? "Start diagnostic full sweep"
            : "Start official fail-fast run"
        appendActivity(
            selectedTarget == .diagnostic
                ? "Viewing the non-certifying diagnostic sweep."
                : "Viewing the official fail-fast run."
        )
        refreshStatus()
    }

    @objc private func copyPressed() {
        guard let status = latestStatus, status.verified else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        if pasteboard.setString(status.summary, forType: .string) {
            appendActivity("Concise verified status copied to the clipboard.")
        } else {
            showSafeError("The status summary could not be copied safely.")
        }
    }

    @objc private func closePressed() {
        window?.performClose(nil)
    }

    @objc private func silenceAlarmPressed() {
        silenceTerminalFailureAlarm(recordActivity: true)
    }

    private func confirm(title: String, message: String, confirmTitle: String) -> Bool {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.addButton(withTitle: confirmTitle)
        alert.addButton(withTitle: "Cancel")
        return alert.runModal() == .alertFirstButtonReturn
    }

    private func runOperation(_ action: HelperAction) {
        guard
            !operationInProgress,
            action != .snapshot,
            action != .diagnosticSnapshot
        else { return }
        operationInProgress = true
        applyButtonState()
        if action == .readiness || action == .diagnosticReadiness {
            appendActivity("Readiness check requested. This cannot train or create the rerun.")
        }
        runHelper(action) { [weak self] exitCode, data in
            guard let self else { return }
            self.operationInProgress = false
            do {
                let result = try decodeOperation(data)
                self.appendActivity(result.message)
                if !result.ok {
                    self.detachedLaunchPending = false
                    self.showSafeError(result.message)
                }
            } catch {
                self.detachedLaunchPending = false
                self.showSafeError(
                    "The local operation returned invalid evidence (exit \(exitCode)); " +
                        "controls remain conservative."
                )
            }
            self.applyButtonState()
            self.scheduleRefresh(after: 0.5)
        }
    }

    private func refreshStatus() {
        guard !statusInProgress else { return }
        statusInProgress = true
        applyButtonState()
        let snapshotAction: HelperAction = selectedTarget == .official
            ? .snapshot
            : .diagnosticSnapshot
        let expectedRun = selectedTarget.expectedRunName
        runHelper(snapshotAction) { [weak self] _, data in
            guard let self else { return }
            self.statusInProgress = false
            do {
                let status = try decodeStatus(data)
                guard status.runName == expectedRun else {
                    throw MonitorDecodeError.invalidShape
                }
                self.latestStatus = status
                if status.runExists {
                    self.detachedLaunchPending = false
                }
                self.render(status)
                let signature = [
                    status.state, status.latestEvent,
                    String(status.eventSequence ?? 0), status.latestMessage,
                    String(status.completedUnits ?? -1), String(status.stopRequested),
                ].joined(separator: "|")
                if signature != self.lastStatusSignature {
                    self.appendActivity(status.summary)
                    self.lastStatusSignature = signature
                }
                self.alertForTerminalFailureIfNeeded(status)
                self.scheduleRefresh(after: status.state == "Running" ? 5 : 15)
            } catch {
                self.latestStatus = nil
                self.detachedLaunchPending = false
                self.stateLabel.stringValue = "Failed — verification refused"
                self.stageLabel.stringValue = "Status evidence is missing, malformed, or mismatched"
                self.appendActivity("Status evidence failed strict validation; controls are disabled.")
                self.scheduleRefresh(after: 15)
            }
            self.applyButtonState()
        }
    }

    private func runHelper(
        _ action: HelperAction,
        completion: @escaping (Int32, Data) -> Void
    ) {
        let root = projectRoot
        let wrapper = controllerWrapper
        DispatchQueue.global(qos: .userInitiated).async {
            let process = Process()
            process.executableURL = wrapper
            process.arguments = [action.rawValue]
            process.currentDirectoryURL = root
            process.standardInput = FileHandle.nullDevice
            let output = Pipe()
            let errors = Pipe()
            process.standardOutput = output
            process.standardError = errors
            var status: Int32 = 5
            var data = Data()
            do {
                try process.run()
                process.waitUntilExit()
                status = process.terminationStatus
                data = output.fileHandleForReading.readDataToEndOfFile()
                _ = errors.fileHandleForReading.readDataToEndOfFile()
                if data.count > maximumHelperOutputBytes {
                    data = Data()
                    status = 5
                }
            } catch {
                data = Data()
                status = 5
            }
            DispatchQueue.main.async {
                completion(status, data)
            }
        }
    }

    private func render(_ status: StatusPayload) {
        stateLabel.stringValue = status.verified ? status.state : "Failed — verification refused"
        if let index = status.stageIndex, let total = status.stageTotal {
            stageLabel.stringValue = "Stage \(index) of \(total) — \(status.currentStage)"
        } else if status.state == "Not started" {
            stageLabel.stringValue = "Stage 0 of 16 — waiting to start"
        } else {
            stageLabel.stringValue = status.currentStage
        }
        overallProgress.doubleValue = status.overallPercent
        overallLabel.stringValue = String(
            format: "%@ — all versions and all 16 stages: %.1f%%",
            status.runName == expectedDiagnosticRunName
                ? "DIAGNOSTIC SWEEP"
                : "OFFICIAL RERUN",
            status.overallPercent
        )
        versionProgress.doubleValue = status.versionPercent
        versionLabel.stringValue = String(
            format: "Current version — %@, stage %d of %d: %.1f%%",
            status.versionName,
            status.versionStageIndex,
            status.versionStageTotal,
            status.versionPercent
        )
        workProgress.doubleValue = status.workPercent
        if let completed = status.completedUnits, let total = status.totalUnits {
            workLabel.stringValue = String(
                format: "Current work: %d of %d (%.1f%%)",
                completed, total, status.workPercent
            )
        } else {
            workLabel.stringValue = "Current work: exact units unavailable"
        }
        let metric: String
        if let name = status.metricName, let value = status.metricValue {
            metric = "\(name) = \(String(format: "%.8g", value))"
        } else {
            metric = "Unavailable"
        }
        detailLabels["message"]?.stringValue = status.latestMessage
        detailLabels["event"]?.stringValue = status.latestEvent
        detailLabels["elapsed"]?.stringValue = formatDuration(status.elapsedSeconds)
        detailLabels["eta"]?.stringValue = formatDuration(status.etaSeconds)
        detailLabels["metric"]?.stringValue = metric
        detailLabels["checkpoint"]?.stringValue = status.latestCheckpoint ?? "Unavailable"
        detailLabels["interruptions"]?.stringValue = String(status.interruptions)
        detailLabels["stop"]?.stringValue = status.stopRequested ? "Pending" : "None"
        detailLabels["commit"]?.stringValue = status.sourceCommit
        detailLabels["run"]?.stringValue = status.runName
        detailLabels["updated"]?.stringValue = status.latestUpdateUtc ?? "Not yet available"
        detailLabels["pipeline"]?.stringValue = status.pipelineStatus
    }

    private func alertForTerminalFailureIfNeeded(_ status: StatusPayload) {
        guard
            status.verified,
            terminalFailureStates.contains(status.state),
            !terminalFailureAlertIssued
        else { return }
        terminalFailureAlertIssued = true
        terminalFailureAlarmActive = true
        applyButtonState()
        appendActivity("Terminal failure detected. Maximum-volume audible alarm started.")
        NSApplication.shared.requestUserAttention(.criticalRequest)
        if let alarm = NSSound(
            contentsOfFile: terminalFailureSoundPath,
            byReference: true
        ) {
            alarm.volume = 1.0
            alarm.loops = true
            terminalFailureAlarm = alarm
            alarm.play()
            let stopItem = DispatchWorkItem { [weak self] in
                self?.silenceTerminalFailureAlarm(recordActivity: false)
            }
            terminalFailureAlarmWorkItems.append(stopItem)
            DispatchQueue.main.asyncAfter(
                deadline: .now() + terminalFailureAlarmDurationSeconds,
                execute: stopItem
            )
        } else {
            for beepIndex in 0 ..< terminalFailureFallbackBeepCount {
                let beepItem = DispatchWorkItem { [weak self] in
                    guard self?.terminalFailureAlarmActive == true else { return }
                    NSSound.beep()
                }
                terminalFailureAlarmWorkItems.append(beepItem)
                DispatchQueue.main.asyncAfter(
                    deadline: .now()
                        + (Double(beepIndex) * terminalFailureFallbackBeepSpacingSeconds),
                    execute: beepItem
                )
            }
        }
    }

    private func silenceTerminalFailureAlarm(recordActivity: Bool) {
        let wasActive = terminalFailureAlarmActive
        terminalFailureAlarm?.stop()
        terminalFailureAlarm = nil
        terminalFailureAlarmWorkItems.forEach { $0.cancel() }
        terminalFailureAlarmWorkItems.removeAll()
        terminalFailureAlarmActive = false
        if recordActivity && wasActive {
            appendActivity("Terminal failure alarm stopped by the owner.")
        }
        applyButtonState()
    }

    private func formatDuration(_ seconds: Double?) -> String {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return "Unavailable" }
        let whole = Int(seconds)
        let hours = whole / 3600
        let minutes = (whole % 3600) / 60
        let remainder = whole % 60
        if hours > 0 {
            return String(format: "%dh %02dm %02ds", hours, minutes, remainder)
        }
        if minutes > 0 {
            return String(format: "%dm %02ds", minutes, remainder)
        }
        return "\(remainder)s"
    }

    private func applyButtonState() {
        let busy = operationInProgress || statusInProgress
        guard let status = latestStatus, status.verified else {
            targetControl.isEnabled = !busy && !detachedLaunchPending
            readinessButton.isEnabled = false
            startButton.isEnabled = false
            refreshButton.isEnabled = !statusInProgress
            stopButton.isEnabled = false
            resumeButton.isEnabled = false
            finderButton.isEnabled = false
            copyButton.isEnabled = false
            silenceAlarmButton.isEnabled = terminalFailureAlarmActive
            closeButton.isEnabled = true
            return
        }
        readinessButton.isEnabled = status.policy.dryRun && !busy && !detachedLaunchPending
        targetControl.isEnabled = !busy && !detachedLaunchPending
        startButton.isEnabled = status.policy.start && status.state == "Not started" &&
            !busy && !detachedLaunchPending
        refreshButton.isEnabled = status.policy.refresh && !statusInProgress
        stopButton.isEnabled = status.policy.stop && status.state == "Running" && !busy
        resumeButton.isEnabled = status.policy.resume && status.state == "Stopped" &&
            !busy && !detachedLaunchPending
        finderButton.isEnabled = status.policy.openFinder && status.runExists && !busy
        copyButton.isEnabled = status.policy.copyStatus && !busy
        silenceAlarmButton.isEnabled = terminalFailureAlarmActive
        closeButton.isEnabled = true
    }

    private func appendActivity(_ message: String) {
        let safe = String(
            message.unicodeScalars.map { scalar in
                scalar.value >= 32 || scalar == "\t" ? Character(String(scalar)) : " "
            }
        ).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !safe.isEmpty else { return }
        activity.append(String(safe.prefix(640)))
        if activity.count > maximumActivityEntries {
            activity.removeFirst(activity.count - maximumActivityEntries)
        }
        while activity.joined(separator: "\n").count > maximumActivityCharacters,
              activity.count > 1
        {
            activity.removeFirst()
        }
        activityView.string = activity.joined(separator: "\n")
        activityView.scrollToEndOfDocument(nil)
    }

    private func scheduleRefresh(after seconds: TimeInterval) {
        refreshTimer?.invalidate()
        refreshTimer = Timer.scheduledTimer(withTimeInterval: seconds, repeats: false) {
            [weak self] _ in self?.refreshStatus()
        }
    }

    private func showSafeError(_ message: String) {
        let alert = NSAlert()
        alert.messageText = "Operation refused safely"
        alert.informativeText = message
        alert.alertStyle = .warning
        alert.runModal()
    }

    func windowShouldClose(_: NSWindow) -> Bool {
        let active = latestStatus?.state == "Running" || detachedLaunchPending
        guard active else { return true }
        return confirm(
            title: "Close the monitor?",
            message: "The Phase 6 pipeline will continue in the background. Closing this " +
                "window does not stop or cancel it. Use Request safe stop first if you want " +
                "the run to stop.",
            confirmTitle: "Close monitor"
        )
    }
}

private final class ApplicationDelegate: NSObject, NSApplicationDelegate {
    private var monitor: MonitorWindowController?

    func applicationDidFinishLaunching(_: Notification) {
        do {
            monitor = try MonitorWindowController.make()
            monitor?.begin()
        } catch {
            let alert = NSAlert()
            alert.messageText = "The local Phase 6 monitor could not open safely."
            alert.alertStyle = .critical
            alert.runModal()
            NSApplication.shared.terminate(nil)
        }
    }

    func applicationShouldTerminateAfterLastWindowClosed(_: NSApplication) -> Bool {
        true
    }
}

let application = NSApplication.shared
private let delegate = ApplicationDelegate()
application.setActivationPolicy(.regular)
application.delegate = delegate
application.run()
