timestamp=2026-05-30T00:00:00+03:00
task_id=task_2026-05-30_beam_width_compare
agent_id=codex
source=user
related_files=[
  "C:/Users/Иван Литвак/Downloads/cayley-beam-gpu-runner (2).ipynb",
  "C:/Users/Иван Литвак/Downloads/cayleypy-megaminx-model1-2556-218-16-4000e.ipynb"
]
intended_action=compare_two_beamsearch_notebooks_widths_2_22_to_2_25_depth_done_8_to_11_gpu_memory_and_runtime
result_status=in_progress
prompt_raw=
"""
Смотри, я хочу потестировать один бимсерч и другой, и сравнить их скорость на ширине 2**22, 2**23, 2**24 и 2**25. Для этого с depth_done=8 нужно по depth_done=11 на каждой ширине оба варианта посчитать и потом табличку вивести в конце

пустота         келипай-бим куда-бим

2**22

Сколько памяти на ГПУ заняло

Сколько длилось вичисление depth_done=8

Сколько длилось вичисление depth_done=9

Сколько длилось вичисление depth_done=10

Сколько длилось вичисление depth_done=11

2**23

Сколько памяти на ГПУ заняло

Сколько длилось вичисление depth_done=8

Сколько длилось вичисление depth_done=9

Сколько длилось вичисление depth_done=10

Сколько длилось вичисление depth_done=11

2**24

Сколько памяти на ГПУ заняло

Сколько длилось вичисление depth_done=8

Сколько длилось вичисление depth_done=9

Сколько длилось вичисление depth_done=10

Сколько длилось вичисление depth_done=11

2**25

Сколько памяти на ГПУ заняло

Сколько длилось вичисление depth_done=8

Сколько длилось вичисление depth_done=9

Сколько длилось вичисление depth_done=10

Сколько длилось вичисление depth_done=11

Если какая-то прога не может запустить на такой ширине - честно отметить в таблице и смотреть дальше.
Для бимсерча на куда моего в конфиге с каждой шириной повишай вот это

STREAM4_ACTIVE_SORT_SLOTS = 1
SHARD_COUNT = 1
SWEEP_CONFIGS = [
    {
...
        "SHARD_COUNT": 1,
        "STREAM4_ACTIVE_SORT_SLOTS": 1,
...
    },
]

Для 2**25 параметри целевие такие
STREAM4_ACTIVE_SORT_SLOTS = 4
SHARD_COUNT = 64

По сути это сравнения нового бимсерча с келипаевским по скорости работи и потреблению памяти
"""
