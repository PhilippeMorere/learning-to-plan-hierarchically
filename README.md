# Learning to Plan Hierarchically (LPH)
Code for paper **Learning to Plan Hierarchically From Curriculum**, _Philippe Morere, Lionel Ott and Fabio Ramos_, published in Robotics and Automation Letters (RA-L), 2019.

LPH is a framework for learning to plan hierarchically in domains with **unknown dynamics**.
Planning is highly hierarchical and generates short plans, by learning abstract skills and reasoning over their effects.
Transition dynamics are automatically learned from interaction data, allowing to build knowledge of skill success conditions.
A curriculum ensures only meaningful abstract skills are learned.

## Dependencies
The code uses python 3.x, all dependencies can be installed with `pip`
* numpy
* sklearn
* gym
* tqdm
* matplotlib
* prettytable

_(optional)_ Pytorch requried to run RRT comparison.


## Running LPH
Examples are provided in the `example` folder. These include running LPH
with the following settings:
* Planning hierarchically, given Primitive AND Abstract skills:
```
python examples/run_hierarchical_given_skills.py
```
* Planning hierarchically and learning Abstract skills, given Primitive skills and their success conditions:
```
python examples/run_hierarchical_learn_skills.py
```
* Planning hierarchically, learning Abstract skills AND learning Primitive skill conditions:
```
python examples/run_hierarchical_learn_skills_conditions.py
```
## Comparisons
Comparisons provided in the paper can be run from scripts in the `comparisons` folder. These include
Q-Learning, Rapidly-Exploring Random Trees (RRT), and Monte-Carlo Tree Search (MCTS).


## Citing our work
If using our work, please cite the journal following paper:
```
@article{morere2019learning,
  title={Learning to Plan Hierarchically from Curriculum},
  author={Morere, Philippe and Ott, Lionel and Ramos, Fabio},
  journal={IEEE Robotics and Automation Letters},
  year={2019},
  publisher={IEEE}
}
```