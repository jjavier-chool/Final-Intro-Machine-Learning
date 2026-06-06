# Final-Intro-Machine-Learning
The final project for CS429. Investigates the main mechanisms and design principles of Reinforcement Learning.

## How to Run
You can run tasks manually using each of:
```console
python task1.py
python task2.py
python task6.py
```
task6.py is where the required testing for task6 is located. Specific tests can be separated according to their order of appearance.
```console
python task6.py 1
python task6.py 2
python task6.py 3
python task6.py 4
python task6.py 5
python task6.py all
```

It may be necessary to downgrade numpy or test in a separate environment with numpy 1.0 due to some jank with numpy 2.0.

Environment method we used:
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
