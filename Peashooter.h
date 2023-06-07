#ifndef _RANDOMSEARCH_H_
#define _RANDOMSEARCH_H_

#include "BaseAgent.h"
#include "HaxBall.h"

#include "Eigen/Dense"
#include "Eigen/Core"

///
/// \brief The RandomSearch class
///
/// A random search agent to improve directly a linear policy.
///
class Peashooter: public BaseAgent
{
public:
  Peashooter();
  ~Peashooter();

  /// Compute the feature vector
  void getFeatVector(const Eigen::Ref<const Eigen::VectorXd> & state,
                     Eigen::Ref<Eigen::VectorXd> m_phi) const;
  void getFeatAgentDir(const Eigen::Ref<const Eigen::Vector2d> & m_vector,
                       int & idx) const;
  void getFeatAgentDis(const Eigen::Ref<const Eigen::Vector2d> & m_vector,
                       int & idx) const;
  void getFeatDirDiff(const Eigen::Ref<const Eigen::VectorXd> & state,
                      const int & idx_agent_dis,
                      int & idx_dir_diff,
                      int & idx_dir_dist,
                      int & idx_dir_case) const;
  void getFeatBallDir(const double & ref_ang,
                      const double & dir_ang,
                      const int & idx_case,
                      int & idx) const;
  void getFeatBallDis(const Eigen::Ref<const Eigen::Vector2d> & m_vector,
                      const int & idx_case,
                      int & idx) const;
  void getFeatBallVel(const Eigen::Ref<const Eigen::VectorXd> & state,
                      int & idx) const;
  void getFeatBallState(const Eigen::Ref<const Eigen::VectorXd> & state,
                        int & idx) const;

  /// Generate action
  void getAction(Eigen::Ref<Eigen::Vector3d> m_action) const;

  /// A linear policy
  void policy(const Eigen::Ref<const Eigen::VectorXd>& state,
              Eigen::Ref<Eigen::VectorXd> action) const override;

  /// A linear policy, which gets the parameters from the outside
  void policy(const Eigen::Ref<const Eigen::VectorXd>& state,
              const Eigen::Ref<const Eigen::MatrixXd>& m_h,
              Eigen::Ref<Eigen::VectorXd> action) const;

  void randomPolicy(Eigen::Ref<Eigen::Vector3d> m_action) const;

  /// Currently hardcoded to go to zero
  double reward(const Eigen::Ref<const Eigen::VectorXd>& s,
                const Eigen::Ref<const Eigen::VectorXd>& action,
                const Eigen::Ref<const Eigen::VectorXd>& s_prime) const override;
  double reward_player(const Eigen::Ref<const Eigen::VectorXd> & state,
                       const Eigen::Ref<const Eigen::VectorXd> & action,
                       const Eigen::Ref<const Eigen::VectorXd> & state_prime) const;
  double reward_shooting(const Eigen::Ref<const Eigen::VectorXd> & state,
                         const Eigen::Ref<const Eigen::VectorXd> & action,
                         const Eigen::Ref<const Eigen::VectorXd> & state_prime) const;

  /// Currently the constant Q-value "42"
  double getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state,
                    const Eigen::Ref<const Eigen::VectorXd>& action) const override;

  /// Currently the constant Q-values "42 ... 42"
  Eigen::VectorXd getQfactor(const Eigen::Ref<const Eigen::VectorXd>& state) const override;

  ///
  /// \brief training
  ///
  /// Trains the policy by performing one iteration of the Cross Entroy Method
  ///
  void td_lambda(int epoch);

private:

  /// A passive and private game instance for getting details about the game (e.g. the goal position for the reward computation)
  /// Do not use this single instance for multithreaded training, as this would mess up its internal state
  /// (That is the reason why this instance is constant)
  const HaxBall m_world;

  static int act_idx;

  /// The covariance matrix used to define the gaussian in Eigen
  Eigen::MatrixXd m_h;
  Eigen::VectorXd m_feature_vector;

  /// Some variable
  Eigen::Vector2d m_real_goal;
  Eigen::Vector2d m_upper_virtual_goal;
  Eigen::Vector2d m_lower_virtual_goal;

public:
  /// Total number of particles for CEM
  static const unsigned int N_TOTAL;

  /// Rollout length
  static const unsigned int TAU;

  /// Discounting for rollouts
  static const double LMDA;
  static const double GAMMA;
  static const double ALPHA;
  static const double EPSILON;
  static const unsigned int N_ACTIONS;

  static const unsigned int N_PHI_1;
  static const unsigned int N_PHI_2;
  static const unsigned int N_PHI_3;
  static const unsigned int N_PHI_4;
  static const unsigned int N_PHI_5;
  static const unsigned int N_PHI_6;
  static const unsigned int N_PHI_7;
};


#endif
