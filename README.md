# Final-Intro-Machine-Learning
The final project for CS429.

## TODO:
- Ensure good documentation comments in each file
- Task 2: Method for enlargening: map1 needs 20x20 to become 40x40
- Task 6: tuning, ensure correctness
- Report on Overleaf

## How to Run
You can run tasks manually using each of:
```console
python task1.py
python task2.py
python task6.py
```

It may be necessary to downgrade numpy or test in a separate environment with numpy 1.0 due to some jank with numpy 2.0.

Environment method I used:
```console
python3 -m venv final-env
source final-env/bin/activate
pip install -r requirements.txt
```
Deactive with:
```console
deactivate
```

Reactivate with:
```console
source final-env/bin/activate
```
