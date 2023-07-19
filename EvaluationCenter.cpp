#include "EvaluationCenter.h"
#include "RewardFunctions.h"

#include <iostream>
#include <fstream>
#include <cmath>

EvaluationCenter::EvaluationCenter(const Peashooter& agent, std::shared_ptr<HaxBall> world, double gamma) :
  m_world(world), m_agent(agent), m_gamma(gamma)
{
  writeHeader();
  writeTrajctory();

  createProbes();
}

EvaluationCenter::EvaluationCenter(const Peashooter& agent, double gamma):
  EvaluationCenter(agent, std::make_shared<HaxBall>(), gamma)
{

}

EvaluationCenter::~EvaluationCenter()
{

}

void EvaluationCenter::evaluate()
{
  Eigen::VectorXd
      state(m_world->getStateDimension()),
      action(m_world->getActionDimension());

  int G_plus, G_minus, act_idx;
  double V, R;
  Eigen::VectorXd r_rec = Eigen::VectorXd::Zero(EvaluationCenter::TAU);
  Eigen::MatrixXd state_rec = Eigen::MatrixXd::Zero(EvaluationCenter::TAU, m_world->getStateDimension());

  std::ofstream file;
  file.open("eval.csv", std::ios::out | std::ios::app);

  // Expected reward for all probes according to agent
  for (int i = 0; i < EvaluationCenter::N; ++i)
  {
    state = m_probes[i];
    m_agent.policy(state, action, act_idx);

    V = m_agent.getQfactor(state, act_idx); // This is not Q but V, since the action is selected according to the policy

    file << V << ",";
  }

  // True Discounted Returns according to rollouts
  for (int i = 0; i < EvaluationCenter::N; ++i)
  {
    std::ostringstream oss;
    oss << "traj_" << i << ".csv";
    std::ofstream file_traj;
    file_traj.open(oss.str(), std::ios::out | std::ios::app);

    rollout(m_probes[i], R, r_rec, G_plus, G_minus, state_rec);

    file << R << "," << G_plus << "," << G_minus;

    for (int k = 0; k < EvaluationCenter::TAU; ++k)
        file_traj << r_rec(k) << "," << state_rec(k, 0) << "," << state_rec(k, 1) << "," << state_rec(k, 2) << "," << state_rec(k, 3) << "," << state_rec(k, 4) << "," << state_rec(k, 5) << std::endl;
    file_traj << "r_" << i << "," << "px_" << i << "," << "py_" << i << "," << "bx_" << i << "," << "by_" << i << "," << "vx_" << i << "," << "vy_" << i << std::endl;

    // The last , must be omitted for .csv format
    if(i < EvaluationCenter::N-1)
    {
       file << ",";
       file_traj.close();
    }
  }

  file << std::endl;

  file.close();
}

void EvaluationCenter::rollout(const Eigen::Ref<const Eigen::VectorXd>& start_state,
                               double & R,
                               Eigen::Ref<Eigen::VectorXd> r_rec,
                               int & G_plus,
                               int & G_minus,
                               Eigen::Ref<Eigen::MatrixXd> state_rec)
{
  Eigen::VectorXd
      state(m_world->getStateDimension()),
      action(m_world->getActionDimension()),
      state_prime(m_world->getStateDimension());

  // Prepare environment
  m_world->reset();
  m_world->setState(start_state);
  double r = 0.0;

  // Recorded values
  R = 0.0;
  r_rec.fill(0.0);
  G_plus = 0;
  G_minus = 0;
  state_rec.fill(0.0);

  // Create rollout
  for(int j = 0; j < EvaluationCenter::TAU; ++j)
  {
    m_world->getState(state);

    m_agent.policy(state, action);

    m_world->step(action);

    m_world->getState(state_prime);

    r = m_agent.reward(state, action, state_prime);

    R += std::pow(m_gamma, j) * r;

    r_rec(j) = r;
    state_rec.row(j) = state;
    if (m_agent.ball_in_goal(state, action, state_prime) > 1)
       G_plus += 1;
    else if (m_agent.ball_in_goal(state, action, state_prime) < -1)
       G_minus += 1;
  }
}

void EvaluationCenter::writeHeader() const
{

  std::ofstream file;

  file.open("eval.csv", std::ios::out);

  // Estimatation for expected reward, i.e., V(s) = Q(s, pi(s))
  for (int i = 0; i < EvaluationCenter::N; ++i)
  {
    file << "V_" << i << ",";
  }

  // True Discounted Returns
  for (int i = 0; i < EvaluationCenter::N; ++i)
  {
    file << "R_" << i << "," << "G_plus_" << i << "," << "G_minus_" << i;

    // The last , must be omitted for .csv format
    if(i < EvaluationCenter::N-1)
       file << ",";
  }

  file << std::endl;

  file.close();
}

void EvaluationCenter::writeTrajctory() const
{
  for (int i = 0; i < EvaluationCenter::N; ++i)
  {
    std::ostringstream oss;
    oss << "traj_" << i << ".csv";

    std::ofstream file;
    file.open(oss.str(), std::ios::out);
    file << "r_" << i << "," << "px_" << i << "," << "py_" << i << "," << "bx_" << i << "," << "by_" << i << "," << "vx_" << i << "," << "vy_" << i << ",";

    file << std::endl;
    file.close();
  }
}

void EvaluationCenter::createProbes()
{
  m_probes.resize(EvaluationCenter::N);

  for (int i = 0; i < m_probes.size(); ++i)
  {
    m_probes[i].resize(m_world->getStateDimension());

    // Quick and dirty way to get a random state from the game
    do {
       m_world->reset();
       m_world->getState(m_probes[i]);
    }
    while ((m_probes[i](0) > 2.0) || (m_probes[i](2) > 2.0));
  }

  m_probes[0] << -1.5, 0.0, 0.0, 0.0, 0.0, 0.0;
  m_probes[1] << -1.0, -1.0, 0.0, 0.0, 0.0, 0.0;
  m_probes[2] << -1.0, 1.0, 0.0, 0.0, 0.0, 0.0;
  m_probes[3] << 1.5, 0.0, 0.0, 0.0, 0.0, 0.0;

  std::ofstream file;
  file.open("probes.csv", std::ios::out);

  file << "p_x,p_y,b_x,b_y,v_x,v_y" << std::endl;

  for (int i = 0; i < m_probes.size(); ++i)
  {
    for (int j = 0; j < m_world->getStateDimension(); ++j)
    {
      file << m_probes[i](j);

      // The last , must be omitted for .csv format
      if(j < m_world->getStateDimension()-1)
         file << ",";
    }

    file << std::endl;

  }

  file.close();
}
