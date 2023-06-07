Applying Reinforcement Learning on Football Game
================================================

This is a project provided by the course ARL 2023 at the Technical University of Munich. Due to the copyright restrictions, I am not allowed to upload the code for the football game environment that created by the lecturer, so you can only browse the code contributed by me, which is however not possible to run without access to this environment. I will also show the current results of the project.

Project Description:
--------------------

This is a smaller version of the online game HaxBall, the target is to train an agent (blue) using reinforcement learning algorithm, protecting my goal (left) and shooting at opponent’s goal (right). As you can see, there is also an opponent (red) that protects his goal. Note that, this project is still in progress.

The challenges for this project could be:
* Constructing a state space or features around a 6-dimensional continuous control problem,
* Constructing a reward function, which produces a good policy,
* Ensuring the flow of information, i.e., proper exploration.

File Description:
-----------------

* `CMakeLists.txt` contains directives and instructions describing the project's source files and targets.
* `main.cpp` is the starting point for program execution. 
* `Peashooter.cpp` is my agent design, describing the state space encoding, action space encoding, reward functions, RL algorithm etc.
* `Peashooter.h` is the header file.

Usage
-----

Assume that you have got the whole environment from the course repository. After cloning, compilation should work in the current directory `RL-Football-Game` with the normal procedure:

```console
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j 8
```

Use an IDE such as QtCreator to open the full project. Your IDE should be able to open the CMakeLists.txt directly:

```console
qtcreator CMakeLists.txt
```

Results
-------

The first video shows the agent can successfully kick the ball into the goal. Note that, the training can be finished only after 16 epochs, which indicates that my solution is suitable for computers with limited performance. The second video shows the decision-making of the agent in some difficult situation. That means, the agent will not kick the ball randomly, instead he will find a feasible shooting direction to score goals, avoiding own goals.