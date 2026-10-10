"""Render a truthful preparation snapshot without changing frozen training code."""
import hashlib
import json
from pathlib import Path
from transport_reasoning.config import write_json
from research.recurrent_filter_sudoku_v1.config import root
from research.recurrent_filter_sudoku_v1.reporting import report

report()
path = Path('REPORT_RECURRENT_SUDOKU_RU.md')
text = path.read_text()
if not list((root() / 'runs/final').glob('**/measurement.json')):
    text = text.replace('Текущая скорость R32 измеряется на тех же ресурсах.',
                        'Текущая скорость R32 будет измерена на тех же ресурсах; этих замеров пока нет.')
    text = text.replace('Forward-only и полный inference, batch 1 и 64, записаны раздельно. В `measurement.json` сохранены повторы, квартили и диагностика.',
                        'Предусмотрены отдельные замеры forward-only и полного inference, batch 1 и 64. После исполнения `measurement.json` будет содержать медианы, квартили, число повторов и диагностику. Сейчас этих измерений нет.')
queue = root() / 'queue/state.json'
if queue.exists():
    value = json.loads(queue.read_text())
    text += '\n## Состояние очереди\n\n'
    text += 'Фактический статус: `' + value['status'] + '`. '
    text += 'GPU-задания новой серии: ' + ', '.join(str(r['job_id']) for r in value['gpu_jobs']) + '. '
    text += 'Подтверждённый дополнительный лимит: ' + ('записан.' if (root()/'queue/budget_grant.json').exists() else 'ещё не задан.') + '\n'
path.write_text(text)
manifest_path = root()/'report/report_manifest.json'
manifest = json.loads(manifest_path.read_text())
manifest['snapshot_renderer_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
write_json(manifest_path, manifest)
