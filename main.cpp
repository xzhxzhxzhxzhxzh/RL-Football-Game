#include <iostream>

#include <QApplication>
#include <QDebug>

#include "HaxBall.h"
#include "HaxBallGui.h"
#include "EvaluationCenter.h"

#include "Peashooter.h"
#include "DummyAgent.h"

void playing(const BaseAgent& agent, int argc, char** argv)
{
  QApplication app(argc, argv);
  HaxBallGui gui(agent);
  gui.show();
  gui.playGame(1.0, -1, true, false);
  app.exec();
}

void render(const BaseAgent& agent, int argc, char** argv)
{
  QApplication app(argc, argv);
  HaxBallGui gui(agent);
  gui.show();
  gui.playGame(1.0, -1, false, false);
  app.exec();
}

int main(int argc, char** argv)
{
  std::cout << "Hello Group Group-2!" << std::endl;

  // My own agent
  Peashooter agent;

  // The evaluation center provides you with some metrics for the progress
  // Results in a .csv files next to the executable
  EvaluationCenter eval(agent, Peashooter::GAMMA);

  // The learning loop
  for (int i = 0; i < 500; ++i)
  {
    std::cout << i << std::endl;

    agent.td_lambda(i);
    if( i % 5 == 1)
    {
      // eval.evaluate();
      render(agent, argc, argv);
    }
  }

  return 0;
}
