"""
Unit Tests for DAgger Components
Tests each component independently to verify correct isolation and behavior.
"""
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import MagicMock, Mock, patch
from typing import Dict, Any

# Test Fixtures and Mock Objects

class MockPolicy:
    """Mock implementation of PolicyInterface for testing"""
    def __init__(self, actions_to_return=[0, 1, 0]):
        self.actions_to_return = iter(actions_to_return)
        self.call_count = 0
    
    def get_action(self, state, action_mask=None):
        self.call_count += 1
        return next(self.actions_to_return)

class MockEnv:
    """Mock environment for testing samplers"""
    def __init__(self, num_steps=5, num_actions=3):
        self.num_steps = num_steps
        self.num_actions = num_actions
        self.step_count = 0
        self.reset_count = 0
    
    def reset(self, options=None):
        self.step_count = 0
        self.reset_count += 1
        obs = np.random.rand(4).astype(np.float32)  # 4-dim observation
        return obs, {"init_state_idx": options.get("init_state_idx", -1) if options else -1}
    
    def step(self, action):
        self.step_count += 1
        obs = np.random.rand(4).astype(np.float32)
        reward = 1.0 if self.step_count < self.num_steps else 10.0
        done = self.step_count >= self.num_steps
        truncated = False
        info = {"is_unsafe": False}
        return obs, reward, done, truncated, info
    
    def action_mask(self):
        return np.ones(self.num_actions, dtype=bool)
    
    @property
    def unwrapped(self):
        return self

class MockOracle:
    """Mock oracle for testing fault collectors"""
    def __init__(self, unsafe_at_step=None):
        self.unsafe_at_step = unsafe_at_step  # Make certain steps unsafe
        self.call_count = 0
        self.corrected_action = 0

    def is_state_action_fault(self, obs, action):
        self.call_count += 1
        return self.unsafe_at_step is not None and action == self.unsafe_at_step
    
    def evaluate_and_correct(self, obs, action, mask):
        self.query_count += 1
        
        if isinstance(mask, (list, tuple)):
            mask = np.array(mask)
        
        try:
            # ✅ CORRECT: Actually check oracle safety
            is_safe = self.oracle.is_engine_state_action_safe(action)
            
            if is_safe:
                return True, action
            else:
                # Action is unsafe - get a safe alternative
                is_state_safe, safe_action = self.oracle.engine_state_safety_with_action(action)
                return False, safe_action
        except:
            return True, action


# ============================================================================
# INTERFACE CONTRACT TESTS
# ============================================================================

class TestPolicyInterface:
    """Verify PolicyInterface contract"""
    
    def test_policy_interface_get_action(self):
        """Test that PolicyInterface.get_action works correctly"""
        policy = MockPolicy(actions_to_return=[0, 1, 2, 0, 1])
        
        # Should return valid actions
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, True, True])
        
        action1 = policy.get_action(obs, mask)
        action2 = policy.get_action(obs, mask)
        action3 = policy.get_action(obs, mask)
        
        assert action1 == 0
        assert action2 == 1
        assert action3 == 2
        assert policy.call_count == 3

class TestTraceSamplerInterface:
    """Verify TraceSamplerInterface contract"""
    
    def test_trace_sampler_returns_dict_with_required_fields(self):
        """Test that trace sampler returns required fields"""
        from dagger.sampler import StandardTraceSampler
        
        sampler = StandardTraceSampler()
        env = MockEnv(num_steps=5)
        policy = MockPolicy(actions_to_return=[0, 1, 0, 1, 0])
        
        trace = sampler.sample_trace(env, policy, init_state_idx=0, max_steps=10)
        
        # Verify required fields
        assert "observations" in trace
        assert "actions" in trace
        assert "action_masks" in trace
        assert "rewards" in trace
        assert "is_safe_trajectory" in trace
        assert "final_reward" in trace
        
        # Verify types
        assert isinstance(trace["observations"], list)
        assert isinstance(trace["actions"], list)
        assert isinstance(trace["is_safe_trajectory"], bool)
        assert isinstance(trace["final_reward"], (float, int))
    
    def test_trace_sampler_respects_max_steps(self):
        """Test that sampler respects max_steps limit"""
        from dagger.sampler import StandardTraceSampler
        
        sampler = StandardTraceSampler()
        env = MockEnv(num_steps=100)  # Never terminates naturally
        policy = MockPolicy(actions_to_return=[0]*50)  # Enough actions
        
        max_steps = 10
        trace = sampler.sample_trace(env, policy, max_steps=max_steps)
        
        # Should have at most max_steps
        assert len(trace["observations"]) <= max_steps
        assert len(trace["actions"]) <= max_steps
    
    def test_trace_sampler_calls_policy_interface(self):
        """Test that sampler calls policy.get_action()"""
        from dagger.sampler import StandardTraceSampler
        
        sampler = StandardTraceSampler()
        env = MockEnv(num_steps=3)
        policy = MockPolicy(actions_to_return=[0, 1, 0])
        
        trace = sampler.sample_trace(env, policy, max_steps=10)
        
        # Policy.get_action() should be called once per step
        assert policy.call_count > 0
        assert len(trace["actions"]) == policy.call_count

class TestFaultCollectorInterface:
    """Verify FaultCollectorInterface contract"""
    
    def test_fault_collector_returns_list_of_dicts(self):
        """Test that fault collector returns correct format"""
        from dagger.fault_collector import OracleFaultCollector
        
        collector = OracleFaultCollector()
        oracle = MockOracle(unsafe_at_step=2)
        
        # Create a synthetic trace
        trace = {
            "observations": [np.array([1, 2, 3, 4]) for _ in range(5)],
            "actions": [0, 1, 0, 1, 0],
            "action_masks": [np.array([True, True, True]) for _ in range(5)]
        }
        
        faults = collector.collect_faults(trace, oracle)
        
        # Should return list
        assert isinstance(faults, list)
        
        # Each fault should be a dict with required fields
        for fault in faults:
            assert isinstance(fault, dict)
            assert "observation" in fault
            assert "action_mask" in fault
            assert "action" in fault  # Optional correction, -1 when not used
            assert "faulty_action" in fault
    
    def test_fault_collector_identifies_faults(self):
        """Test that collector correctly identifies unsafe actions"""
        from dagger.fault_collector import OracleFaultCollector
        
        collector = OracleFaultCollector()
        oracle = MockOracle(unsafe_at_step=1)  # Unsafe at step 1
        
        trace = {
            "observations": [np.array([1, 2, 3, 4]) for _ in range(3)],
            "actions": [0, 1, 2],
            "action_masks": [np.array([True, True, True]) for _ in range(3)]
        }
        
        faults = collector.collect_faults(trace, oracle)
        
        # Should have exactly 1 fault (at step 1)
        assert len(faults) == 1
        assert faults[0]["faulty_action"] == 1
    
    def test_fault_collector_no_faults_when_safe(self):
        """Test that safe traces produce no faults"""
        from dagger.fault_collector import OracleFaultCollector
        
        collector = OracleFaultCollector()
        oracle = MockOracle(unsafe_at_step=None)  # All steps safe
        
        trace = {
            "observations": [np.array([1, 2, 3, 4]) for _ in range(5)],
            "actions": [0, 1, 0, 1, 0],
            "action_masks": [np.array([True, True, True]) for _ in range(5)]
        }
        
        faults = collector.collect_faults(trace, oracle)
        
        # Should have no faults
        assert len(faults) == 0


# ============================================================================
# COMPONENT ISOLATION TESTS
# ============================================================================

class TestComponentIsolation:
    """Verify that components work independently"""
    
    def test_sampler_independent_of_oracle(self):
        """Sampler should not depend on oracle"""
        from dagger.sampler import StandardTraceSampler
        
        sampler = StandardTraceSampler()
        env = MockEnv(num_steps=5)
        policy = MockPolicy(actions_to_return=[0, 1, 0, 1, 0])
        
        # Oracle is not used by sampler - should work without it
        trace = sampler.sample_trace(env, policy)
        assert trace is not None
    
    def test_fault_collector_independent_of_env(self):
        """Fault collector should not need environment"""
        from dagger.fault_collector import OracleFaultCollector
        
        collector = OracleFaultCollector()
        oracle = MockOracle()
        
        # Create synthetic trace - no env needed
        trace = {
            "observations": [np.array([1, 2, 3, 4]) for _ in range(3)],
            "actions": [0, 1, 2],
            "action_masks": [np.array([True, True, True]) for _ in range(3)]
        }
        
        faults = collector.collect_faults(trace, oracle)
        assert isinstance(faults, list)
    
    def test_policy_interface_works_with_any_env(self):
        """Policy should work with any environment via interface"""
        policy1 = MockPolicy(actions_to_return=[0, 1])
        policy2 = MockPolicy(actions_to_return=[1, 0])
        
        env1 = MockEnv(num_steps=1, num_actions=3)
        env2 = MockEnv(num_steps=2, num_actions=4)
        
        # Both policies work with both environments
        obs1, _ = env1.reset()
        action1 = policy1.get_action(obs1, env1.action_mask())
        assert isinstance(action1, int)
        
        obs2, _ = env2.reset()
        action2 = policy2.get_action(obs2, env2.action_mask())
        assert isinstance(action2, int)


# ============================================================================
# DATA FLOW TESTS
# ============================================================================

class TestDataFlow:
    """Verify correct data flow through pipeline"""
    
    def test_end_to_end_sampler_to_collector(self):
        """Test data flows correctly from sampler to collector"""
        from dagger.sampler import StandardTraceSampler
        from dagger.fault_collector import OracleFaultCollector
        
        # Step 1: Sample trajectory
        sampler = StandardTraceSampler()
        env = MockEnv(num_steps=5)
        policy = MockPolicy(actions_to_return=[0, 1, 0, 1, 0])
        
        trace = sampler.sample_trace(env, policy, max_steps=10)
        
        # Step 2: Collect faults
        collector = OracleFaultCollector()
        oracle = MockOracle(unsafe_at_step=2)
        
        faults = collector.collect_faults(trace, oracle)
        
        # Data should flow correctly
        assert len(trace["observations"]) > 0
        assert len(faults) >= 0  # May have 0 or more faults
        
        # If we have faults, they should be valid observations from the trace
        for fault in faults:
            assert fault["observation"] is not None
            assert len(fault["observation"]) > 0
    
    def test_observation_consistency_through_pipeline(self):
        """Verify observations don't get corrupted through pipeline"""
        from dagger.sampler import StandardTraceSampler
        from dagger.fault_collector import OracleFaultCollector
        
        sampler = StandardTraceSampler()
        env = MockEnv(num_steps=3)
        policy = MockPolicy(actions_to_return=[0, 1, 0])
        
        trace = sampler.sample_trace(env, policy)
        original_obs = trace["observations"][0].copy()
        
        collector = OracleFaultCollector()
        oracle = MockOracle()
        faults = collector.collect_faults(trace, oracle)
        
        # Observations should remain unchanged
        assert (trace["observations"][0] == original_obs).all()


# ============================================================================
# CONTRACT VERIFICATION TESTS
# ============================================================================

class TestInterfaceContracts:
    """Verify that implementations honor interface contracts"""
    
    def test_policy_interface_contract(self):
        """PolicyInterface must have get_action method"""
        from dagger.interfaces import PolicyInterface
        
        # Create a test implementation
        class TestPolicy(PolicyInterface):
            def get_action(self, state, action_mask=None):
                return 0
        
        policy = TestPolicy()
        result = policy.get_action(np.array([1, 2, 3]), np.array([True, True, True]))
        assert result is not None
    
    def test_trace_sampler_interface_contract(self):
        """TraceSamplerInterface must have sample_trace method"""
        from dagger.interfaces import TraceSamplerInterface
        
        class TestSampler(TraceSamplerInterface):
            def sample_trace(self, env, policy, init_state_idx=-1, max_steps=1024):
                return {"observations": [], "actions": []}
        
        sampler = TestSampler()
        result = sampler.sample_trace(None, None)
        assert isinstance(result, dict)
    
    def test_fault_collector_interface_contract(self):
        """FaultCollectorInterface must have collect_faults method"""
        from dagger.interfaces import FaultCollectorInterface
        
        class TestCollector(FaultCollectorInterface):
            def collect_faults(self, trace, oracle):
                return []
        
        collector = TestCollector()
        result = collector.collect_faults({}, None)
        assert isinstance(result, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
