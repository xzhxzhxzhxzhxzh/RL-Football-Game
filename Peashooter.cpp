#include "Peashooter.h"

#include <cmath>
#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>
#include <random>

#include <QDebug>

#include <omp.h>

#include "RewardFunctions.h"

const unsigned int Peashooter::N_TOTAL = 450000;
const unsigned int Peashooter::TAU = 120;
const double Peashooter::GAMMA = 0.9;
const double Peashooter::ALPHA = 0.0005;
const double Peashooter::LMDA = 0.9;
const double Peashooter::EPSILON = 0.2;
const unsigned int Peashooter::N_ACTIONS = 9;

const unsigned int Peashooter::N_PHI_1 = 12;
const unsigned int Peashooter::N_PHI_2 = 16;
const unsigned int Peashooter::N_PHI_3 = 38;
const unsigned int Peashooter::N_PHI_4 = 20;
const unsigned int Peashooter::N_PHI_5 = 3;
const unsigned int Peashooter::N_PHI_6 = 6;
const unsigned int Peashooter::N_PHI_7 = 3;

int Peashooter::act_idx;

Peashooter::Peashooter()
{
  Peashooter::act_idx = Peashooter::N_ACTIONS;

  m_feature_vector.resize(Peashooter::N_PHI_1 + Peashooter::N_PHI_2 + Peashooter::N_PHI_3 + Peashooter::N_PHI_4 + Peashooter::N_PHI_5 + Peashooter::N_PHI_6 + Peashooter::N_PHI_7);
  m_h.resize(m_feature_vector.size(), Peashooter::N_ACTIONS);
  m_h.fill(0.0);

  // Initialize some goal position
  Eigen::Vector2d width;
  width << 0.0, m_world.getSize().bottomRight().y();
  m_real_goal << m_world.getGoalRight().center().x(), m_world.getGoalRight().center().y();
  m_upper_virtual_goal = m_real_goal - 2 * width;
  m_lower_virtual_goal = m_real_goal + 2 * width;
}
Peashooter::~Peashooter()
{

}

void Peashooter::getFeatVector(const Eigen::Ref<const Eigen::VectorXd> & state,
                            Eigen::Ref<Eigen::VectorXd> m_phi) const
{
  // Initialize feature vectors
  m_phi.fill(0.0);

  // Initialize feature indexes
  int idx_1 = 0;
  int idx_2 = 0;
  int idx_3 = 0;
  int idx_4 = 0;
  int idx_5 = 0;
  int idx_6 = 0;
  int idx_7 = 0;

  // Get feature indexes
  Eigen::Vector2d m_agent_dist = state.segment(0, 2) - state.segment(2, 2);
  getFeatAgentDir(m_agent_dist, idx_1);
  getFeatAgentDis(m_agent_dist, idx_2);
  getFeatDirDiff(state, idx_2, idx_3, idx_4, idx_7);
  getFeatBallState(state, idx_5);
  getFeatBallVel(state, idx_6);

  // Construct feature vector
  m_phi(idx_1) = 1.0;
  m_phi(Peashooter::N_PHI_1 + idx_2) = 1.0;
  m_phi(Peashooter::N_PHI_1 + Peashooter::N_PHI_2 + idx_3) = 1.0;
  m_phi(Peashooter::N_PHI_1 + Peashooter::N_PHI_2 + Peashooter::N_PHI_3 + idx_4) = 1.0;
  m_phi(Peashooter::N_PHI_1 + Peashooter::N_PHI_2 + Peashooter::N_PHI_3 + Peashooter::N_PHI_4 + idx_5) = 1.0;
  m_phi(Peashooter::N_PHI_1 + Peashooter::N_PHI_2 + Peashooter::N_PHI_3 + Peashooter::N_PHI_4 + Peashooter::N_PHI_5 + idx_6) = 1.0;
  m_phi(Peashooter::N_PHI_1 + Peashooter::N_PHI_2 + Peashooter::N_PHI_3 + Peashooter::N_PHI_4 + Peashooter::N_PHI_5 + Peashooter::N_PHI_6 + idx_7) = 1.0;
}

void Peashooter::getFeatAgentDir(const Eigen::Ref<const Eigen::Vector2d> & m_vector,
                              int & idx) const
{
  // step = 30, size = 12
  double angle = std::atan2(m_vector(1), m_vector(0)) * 180 / M_PI;
  // +++released later+++

  // -15 : 15  -> 0
  //  15 : 45  -> 1
  //  45 : 75  -> 2
  //  75 : 105 -> 3
  //  105: 135 -> 4
  //  135: 165 -> 5
  //  165:-165 -> 6
  // -165:-135 -> 7
  // -135:-105 -> 8
  // -105:-75  -> 9
  // -75 :-45  -> 10
  // -45 :-15  -> 11
}

void Peashooter::getFeatAgentDis(const Eigen::Ref<const Eigen::Vector2d> & m_vector,
                              int & idx) const
{
  // dist < 8.95
  // y = exp(0.128 * x) - 1
  // size = 16

  // +++released later+++

  // < 0.5 -> 0
  //       -> 1
  //       -> 2
  //       -> 3
  // ...
  // 8.95  -> 15
}

void Peashooter::getFeatDirDiff(const Eigen::Ref<const Eigen::VectorXd> & state,
                             const int & idx_agent_dis,
                             int & idx_dir_diff,
                             int & idx_dir_dist,
                             int & idx_dir_case) const
{
  Eigen::Vector2d m_ball_pos = state.segment(2, 2);

  Eigen::Vector2d m_ball_dir;
  double ball_vel = state.segment(4, 2).norm();
  if (ball_vel >= 1e-6)
      m_ball_dir = state.segment(4, 2);
  else
      m_ball_dir = m_ball_pos - state.segment(0, 2);

  double dir_ang = std::atan2(m_ball_dir(1), m_ball_dir(0)) * 180 / M_PI;
  if (dir_ang < 0.0) dir_ang += 360;

  // +++released later+++
}

void Peashooter::getFeatBallDir(const double & ref_ang,
                             const double & dir_ang,
                             const int & idx_case,
                             int & idx) const
{
  // -154 <= ang_diff <= +154
  // y = 100 (exp(0.0491 * x) - 1)
  // size = (19 + 19) = 38
  const int size_single = 38;
  double ang_diff = dir_ang - ref_ang;
  // +++released later+++

  // -154  :-142   -> 0
  // ...
  // -10.31:-5.03  -> 17
  // - 5.03: 0     -> 18
  //   0   : 5.03  -> 19
  //   5.03: 10.31 -> 20
  // ...
  //   142 : 154   -> 37
}

void Peashooter::getFeatBallDis(const Eigen::Ref<const Eigen::Vector2d> & m_vector,
                             const int & idx_case,
                             int & idx) const
{
  // dist < 9.69
  // y = 2 * (exp(0.091 * x) - 1)
  // size = 20
  const int size_single = 20;
  idx = std::round(std::log(m_vector.norm() / 2.0 + 1) / 0.091);
  idx = idx_case * size_single + idx;
  // 0        -> 0
  // 0.19     -> 1
  // ...
  // 9.69     -> 19
}

void Peashooter::getFeatBallVel(const Eigen::Ref<const Eigen::VectorXd> & state,
                             int & idx) const
{
  // dist < 8.5
  // y = 0.1 * (exp(0.87 * x) - 1)
  // size = 4 + 1 + 1 = 6
  double ball_vel = state.segment(4, 2).norm();
  if (ball_vel < 1e-6) idx = 5;
  else if (ball_vel > 2.0) idx = 4;
  else idx = std::round(std::log(ball_vel / 0.1 + 1) / 0.87);
  // 0+       -> 0
  // ...
  // 2        -> 3
  // 2:8.5    -> 4
  // zero vel -> 5
}

void Peashooter::getFeatBallState(const Eigen::Ref<const Eigen::VectorXd> & state,
                                  int & idx) const
{
  double ball_vel = state.segment(4, 2).norm();
  double agent_dist = (state.segment(0, 2) - state.segment(2, 2)).norm();
  if (ball_vel >= 1e-6) idx = 0;
  else if (agent_dist >= 0.5) idx = 1;
  else idx = 2;
}

void Peashooter::getAction(Eigen::Ref<Eigen::Vector3d> m_action) const
{
  // Stay put and shooting
  if (Peashooter::act_idx == 8) m_action(2) = 1.0;
  else
  {
      double radians = Peashooter::act_idx * 45.0 * M_PI / 180.0;
      const double vel_norm = 1.0;

      double val_x = vel_norm * std::sin(radians);
      double val_y = vel_norm * std::cos(radians);

      m_action(0) = val_x;
      m_action(1) = val_y;
  }

  // 0  -> 0
  // 1  -> 30
  // 2  -> 60
  // 3  -> 90
  // 4  -> 120
  // 5  -> 150
  // 6  -> 180
  // 7  -> 210
  // 8  -> 240
  // 9  -> 270
  // 10 -> 300
  // 11 -> 330
}

void Peashooter::policy(const Eigen::Ref<const Eigen::VectorXd>& state,
                          Eigen::Ref<Eigen::VectorXd> action) const
{
  policy(state, m_h, action);

  return;
}

void Peashooter::policy(const Eigen::Ref<const Eigen::VectorXd>& state,
                        const Eigen::Ref<const Eigen::MatrixXd>& m_h,
                        Eigen::Ref<Eigen::VectorXd> action) const
{
  // Get feature state
  Eigen::VectorXd m_phi(m_feature_vector.size());
  getFeatVector(state, m_phi);

  Eigen::VectorXd m_result = m_h.transpose() * m_phi;

  Eigen::Index act_index;
  m_result.maxCoeff(& act_index);
  Peashooter::act_idx = static_cast<int>(act_index);

  getAction(action);
}

void Peashooter::randomPolicy(Eigen::Ref<Eigen::Vector3d> m_action) const
{
  std::random_device rd;
  std::mt19937 gen(rd());

  std::uniform_int_distribution<int> dis(0, Peashooter::N_ACTIONS - 1);

  Peashooter::act_idx = dis(gen);

  getAction(m_action);
}

double Peashooter::reward(const Eigen::Ref<const Eigen::VectorXd>& state,
                       const Eigen::Ref<const Eigen::VectorXd>& action,
                       const Eigen::Ref<const Eigen::VectorXd>& state_prime) const
{
  double reward_t = reward_player(state, action, state_prime)
                    + reward_shooting(state, action, state_prime)
                    + Reward::ball_in_goal(state, action, state_prime);

  return reward_t;
}

double Peashooter::reward_player(const Eigen::Ref<const Eigen::VectorXd> & state,
                              const Eigen::Ref<const Eigen::VectorXd> & action,
                              const Eigen::Ref<const Eigen::VectorXd> & state_prime) const
{
  double reward_i = Reward::distance_player_ball_dense(state, action, state_prime);

  if (reward_i > -0.5) return 0.0;
  else return reward_i;
}

double Peashooter::reward_shooting(const Eigen::Ref<const Eigen::VectorXd> & state,
                                const Eigen::Ref<const Eigen::VectorXd> & action,
                                const Eigen::Ref<const Eigen::VectorXd> & state_prime) const
{
  if (action(2) == 1.0)
  {
      if (Reward::distance_player_ball_dense(state, action, state_prime) > -0.5)
      {
        if (state_prime(4) >= 1e-6) return 10.0;
        else if (state_prime(4) < 0) return -10.0;
        else return 0.0;
      }
      else
      {
        return 0.0;
      }
  }
  else return 0.0;
}

double Peashooter::getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state,
                           const Eigen::Ref<const Eigen::VectorXd>& action) const
{
  return 42.0f;
}

Eigen::VectorXd Peashooter::getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state) const
{
  Eigen::Vector4d Q;  // I took 4 components without particular reason

  Q.fill(42.0);

  return Q;
}

void Peashooter::td_lambda(int epoch)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

#pragma omp parallel for
  for (int i = 0; i < Peashooter::N_TOTAL; ++i)
  {
    // One step reward and accumulator for discounted return
    double r = 0.0;

    // Initialises the environment randomly
    HaxBall env;

    // Variables to store the s,a,s' tuple, one copy per worker thread
    Eigen::VectorXd
        state(env.getStateDimension()),
        m_phi(m_feature_vector.size()),
        m_action(env.getActionDimension()),
        state_prime(env.getStateDimension()),
        m_phi_prime(m_feature_vector.size());

    Eigen::VectorXd e_trace = Eigen::VectorXd::Zero(m_feature_vector.size());

    Eigen::VectorXd
        state_stuck_1(env.getStateDimension()),
        state_stuck_2(env.getStateDimension()),
        state_stuck_3(env.getStateDimension());
    int stuck = 0;

    // Create rollout (finite horizon approximation for infinite horizon, choose TAU long or GAMMA small enough
    for(int j = 0; j < Peashooter::TAU ; ++j)
    {
        env.getState(state);
        getFeatVector(state, m_phi);

        if (distribution(gen) < Peashooter::EPSILON * std::exp(-0.06 * (epoch - 1)))
            randomPolicy(m_action);
        else
            policy(state, m_h, m_action);

        env.step(m_action);

        env.getState(state_prime);
        getFeatVector(state_prime, m_phi_prime);

        r = reward(state, m_action, state_prime);

        // Q-Learning with LVFA and TD(Lambda)
        e_trace = m_phi + Peashooter::LMDA * Peashooter::GAMMA * e_trace;
        m_h.col(Peashooter::act_idx) = m_h.col(Peashooter::act_idx)
                           + Peashooter::ALPHA * e_trace * (r
                                                         + Peashooter::GAMMA * m_h.col(Peashooter::act_idx).dot(m_phi_prime)
                                                         - m_h.col(Peashooter::act_idx).dot(m_phi));

        if (j == 0)
        {
            state_stuck_1 = state;
        }
        else if (j == 1)
        {
            state_stuck_2 = state;
            state_stuck_3 = state_prime;

        }
        else
        {
            state_stuck_1 = state_stuck_2;
            state_stuck_2 = state;
            state_stuck_3 = state_prime;
        }

        if (j > 1)
        {
            if ((state_stuck_2 - state_stuck_3).lpNorm<Eigen::Infinity>() < 1e-6
                || (state_stuck_1 - state_stuck_3).lpNorm<Eigen::Infinity>() < 1e-6)
            {
                stuck += 1;
                if (stuck > 1) break;
            }
            else
            {
                stuck = 0;
            }
        }
    }
  }

  std::cout << "Matrix mat:" << std::endl;
  //std::cout << m_h << std::endl;

}
