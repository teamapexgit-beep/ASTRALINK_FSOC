import sys, time, traceback
sys.path.insert(0, '.')
import io
from PySide6.QtWidgets import QApplication

src = io.open('gui.py', encoding='utf-8').read()
src = src.replace('sys.exit(app.exec())', 'pass')
g = {'__name__': '__main__'}
exec(compile(src, 'gui.py', 'exec'), g)
FSOCWindow = g['FSOCWindow']

app = QApplication.instance() or QApplication(sys.argv)
if QApplication.instance() is None:
    app = QApplication(sys.argv)

w = FSOCWindow()
w.show()

from auth_config import APP_PASSWORD
w.auth_input.setText(APP_PASSWORD)
w._authenticate()

deadline = time.time() + 26
ticks = 0
errors = []
while time.time() < deadline:
    try:
        w.update_simulation()
    except Exception as e:
        errors.append((type(e).__name__, str(e)))
        traceback.print_exc()
        break
    ticks += 1
    time.sleep(0.02)

print('TICKS', ticks)
print('tracker state', str(w.tracker.state))
print('metrics_started', repr(getattr(w, 'metrics_started', 'ABSENT')))
print('fps', repr(round(getattr(w, 'fps', 0), 2)))
print('error_graph series count', repr(len(w.error_graph.series)))
print('motion_graph series count', repr(len(w.motion_graph.series)))
print('last error', repr(w.error_history[-1] if w.error_history else None))
print('authentication', repr(w._authenticated))
print('last snapshot keys', repr(sorted(w._last_report_snapshot.keys()) if w._last_report_snapshot else None))
print('t_state', str(w.t_state.text()))
print('fps label', str(w.t_fps.text()))
print('error label', str(w.t_error.text()))
print('state label', str(w.t_state.text()))
print('link label ascii', str(w.link_readiness_label.text().encode('ascii', 'replace')))
print('footer ascii', str(w.footer_status.text().encode('ascii', 'replace')))
print('sim run', repr(w.simulation_running))
print('SIMULATION_ERROR', str(w.footer_status.text().encode('ascii', 'replace')))
print('errors', errors)
PY