/* extension.js - Ara Status Indicator
 *
 * GNOME Shell extension that listens to Ara Telemetry Fusion (ATF)
 * and reconfigures the cockpit based on Ara's realtime state.
 *
 * D-Bus Interface:
 *   Service: org.scythe.Ara
 *   Object: /org/scythe/Ara/Telemetry
 *   Interface: org.scythe.Ara.Telemetry
 *   Signal: StatusUpdate(payload_json)
 *
 * Installation:
 *   cp -r ara-status@scythe.dev ~/.local/share/gnome-shell/extensions/
 *   gnome-extensions enable ara-status@scythe.dev
 *   Alt+F2, type 'r', Enter (restart GNOME Shell on X11)
 */

import Gio from 'gi://Gio';
import GLib from 'gi://GLib';
import St from 'gi://St';
import Main from 'resource:///org/gnome/shell/ui/main.js';
import PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';
import * as PopupMenu from 'resource:///org/gnome/shell/ui/popupMenu.js';

const BUS_NAME = 'org.scythe.Ara';
const OBJ_PATH = '/org/scythe/Ara/Telemetry';
const IFACE    = 'org.scythe.Ara.Telemetry';

const HOME_DIR = GLib.get_home_dir();
const MODE_SCRIPT = GLib.build_filenamev([HOME_DIR, 'bin', 'ara_mode.sh']);

export default class AraStatusExtension {
    constructor() {
        this._indicator = null;
        this._proxy = null;
        this._signalId = 0;

        this._state = 'IDLE';
        this._lastMode = '';
        this._metrics = {};
    }

    enable() {
        log('[AraStatus] Enabling extension...');

        // Create panel indicator
        this._indicator = new PanelMenu.Button(0.0, 'AraStatus', false);

        // Create icon
        this._icon = new St.Icon({
            icon_name: 'audio-input-microphone-symbolic',
            style_class: 'system-status-icon ara-status-icon-idle',
        });

        this._indicator.add_child(this._icon);

        // Create popup menu for manual controls
        this._createMenu();

        // Add to panel (position 1 from right, in 'right' box)
        Main.panel.addToStatusArea('AraStatus', this._indicator, 1, 'right');

        // Connect to D-Bus
        this._connectDBus();

        log('[AraStatus] Extension enabled');
    }

    disable() {
        log('[AraStatus] Disabling extension...');

        // Disconnect D-Bus
        if (this._proxy && this._signalId) {
            this._proxy.disconnect(this._signalId);
            this._signalId = 0;
        }
        this._proxy = null;

        // Destroy indicator
        if (this._indicator) {
            this._indicator.destroy();
            this._indicator = null;
        }

        log('[AraStatus] Extension disabled');
    }

    _createMenu() {
        // Manual mode switches
        let cruiseItem = new PopupMenu.PopupMenuItem('⚪ Cruise Mode');
        cruiseItem.connect('activate', () => this._triggerMode('cruise'));
        this._indicator.menu.addMenuItem(cruiseItem);

        let flightItem = new PopupMenu.PopupMenuItem('🔵 Flight Mode');
        flightItem.connect('activate', () => this._triggerMode('flight'));
        this._indicator.menu.addMenuItem(flightItem);

        let battleItem = new PopupMenu.PopupMenuItem('🔴 Battle Mode');
        battleItem.connect('activate', () => this._triggerMode('battle'));
        this._indicator.menu.addMenuItem(battleItem);

        this._indicator.menu.addMenuItem(new PopupMenu.PopupSeparatorMenuItem());

        // Metrics display
        this._metricsItem = new PopupMenu.PopupMenuItem('Metrics: Waiting...', {
            reactive: false,
        });
        this._indicator.menu.addMenuItem(this._metricsItem);
    }

    _connectDBus() {
        log('[AraStatus] Connecting to D-Bus...');

        try {
            this._proxy = Gio.DBusProxy.new_for_bus_sync(
                Gio.BusType.SESSION,
                Gio.DBusProxyFlags.NONE,
                null,  // GDBusInterfaceInfo
                BUS_NAME,
                OBJ_PATH,
                IFACE,
                null   // cancellable
            );

            this._signalId = this._proxy.connect('g-signal',
                (proxy, sender, signalName, parameters) => {
                    if (signalName === 'StatusUpdate') {
                        let payloadJson = parameters.get_child_value(0).deepUnpack();
                        this._handleStatusUpdate(payloadJson);
                    }
                }
            );

            log('[AraStatus] D-Bus connected successfully');
        } catch (e) {
            logError(e, '[AraStatus] D-Bus connection failed');
            log('[AraStatus] Hint: Is ara_telemetry_daemon.py running?');
        }
    }

    _handleStatusUpdate(payloadJson) {
        let payload;
        try {
            payload = JSON.parse(payloadJson);
        } catch (e) {
            logError(e, '[AraStatus] Failed to parse JSON payload');
            return;
        }

        // Update state
        let state = payload.State || 'IDLE';
        this._state = state;
        this._metrics = payload;

        // Update icon style
        this._updateIconStyle();

        // Trigger mode script if state changed
        this._autoSwitchMode(state);

        // Update metrics display in menu
        this._updateMetricsDisplay();
    }

    _updateIconStyle() {
        // Remove all state classes
        this._icon.remove_style_class_name('ara-status-icon-idle');
        this._icon.remove_style_class_name('ara-status-icon-flight');
        this._icon.remove_style_class_name('ara-status-icon-battle');
        this._icon.remove_style_class_name('ara-status-icon-critical');

        // Apply new class based on state
        switch (this._state) {
        case 'THINKING':
        case 'PROCESSING':
            this._icon.add_style_class_name('ara-status-icon-flight');
            break;
        case 'SPEAKING':
            this._icon.add_style_class_name('ara-status-icon-battle');
            break;
        case 'CRITICAL':
            this._icon.add_style_class_name('ara-status-icon-critical');
            break;
        default:
            this._icon.add_style_class_name('ara-status-icon-idle');
        }
    }

    _autoSwitchMode(state) {
        let targetMode;

        // Map Ara state to cockpit mode
        switch (state) {
        case 'THINKING':
        case 'PROCESSING':
            targetMode = 'flight';
            break;
        case 'SPEAKING':
            targetMode = 'battle';
            break;
        case 'CRITICAL':
            targetMode = 'battle';  // Or 'panic' if you add that
            break;
        default:
            targetMode = 'cruise';
        }

        // Only trigger if mode actually changed
        if (targetMode !== this._lastMode) {
            log(`[AraStatus] Auto-switching to ${targetMode} mode (state: ${state})`);
            this._triggerMode(targetMode);
            this._lastMode = targetMode;
        }
    }

    _triggerMode(mode) {
        log(`[AraStatus] Triggering mode: ${mode}`);

        // Execute mode script asynchronously
        try {
            GLib.spawn_command_line_async(`${MODE_SCRIPT} ${mode}`);
        } catch (e) {
            logError(e, `[AraStatus] Failed to execute mode script: ${MODE_SCRIPT}`);
        }
    }

    _updateMetricsDisplay() {
        if (!this._metricsItem) return;

        let m = this._metrics;
        let text = `State: ${m.State || 'N/A'}\n` +
                   `GPU: ${(m.GPU_Load_Percent || 0).toFixed(1)}%\n` +
                   `FPGA: ${(m.FPGA_Latency_ms || 0).toFixed(2)} ms\n` +
                   `Target Met: ${m.Target_Lat_Met ? '✓' : '✗'}\n` +
                   `Personality: ${m.Personality_Mode || 'unknown'}`;

        this._metricsItem.label.set_text(text);
    }
}
