import Gio from 'gi://Gio';
import St from 'gi://St';
import GLib from 'gi://GLib';
import Main from 'resource:///org/gnome/shell/ui/main.js';
import PanelMenu from 'resource:///org/gnome/shell/ui/panelMenu.js';

const BUS_NAME = 'org.scythe.Ara';
const OBJ_PATH = '/org/scythe/Ara/Telemetry';
const IFACE    = 'org.scythe.Ara.Telemetry';

export default class AraStatusExtension {
    constructor() {
        this._indicator = null;
        this._proxy = null;
        this._signalId = 0;
        this._state = 'IDLE';
    }

    enable() {
        this._indicator = new PanelMenu.Button(0.0, 'AraStatus');
        this._icon = new St.Icon({
            icon_name: 'audio-input-microphone-symbolic',
            style_class: 'system-status-icon ara-status-icon-idle',
        });
        this._indicator.add_child(this._icon);
        Main.panel.addToStatusArea('AraStatus', this._indicator, 1, 'right');

        this._proxy = Gio.DBusProxy.new_for_bus_sync(
            Gio.BusType.SESSION,
            Gio.DBusProxyFlags.NONE,
            null,
            BUS_NAME,
            OBJ_PATH,
            IFACE,
            null
        );

        this._signalId = this._proxy.connect('g-signal',
            (proxy, sender, signalName, params) => {
                if (signalName === 'StatusUpdate') {
                    const payloadJson = params.get_child_value(0).deepUnpack();
                    this._handleStatusUpdate(payloadJson);
                }
            }
        );
    }

    disable() {
        if (this._proxy && this._signalId) {
            this._proxy.disconnect(this._signalId);
        }
        this._proxy = null;
        this._signalId = 0;

        if (this._indicator) {
            this._indicator.destroy();
            this._indicator = null;
        }
    }

    _handleStatusUpdate(payloadJson) {
        let payload;
        try {
            payload = JSON.parse(payloadJson);
        } catch (e) {
            logError(e, 'AraStatus: JSON parse failed');
            return;
        }

        const state = payload.State || 'IDLE';
        if (state === this._state)
            return;

        this._state = state;
        this._updateIconStyle();
        this._triggerModeScript(state);
    }

    _updateIconStyle() {
        ['ara-status-icon-idle',
         'ara-status-icon-flight',
         'ara-status-icon-battle',
         'ara-status-icon-critical'].forEach(cls => {
            this._icon.remove_style_class_name(cls);
        });

        switch (this._state) {
        case 'THINKING':
        case 'PROCESS':
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

    _triggerModeScript(state) {
        let mode;
        switch (state) {
        case 'THINKING':
        case 'PROCESS':
            mode = 'flight';
            break;
        case 'SPEAKING':
            mode = 'battle';
            break;
        case 'CRITICAL':
            mode = 'battle';
            break;
        default:
            mode = 'cruise';
        }

        const cmd = GLib.build_filenamev([GLib.get_home_dir(), 'bin', 'ara_mode.sh']);
        GLib.spawn_command_line_async(`${cmd} ${mode}`);
    }
}
