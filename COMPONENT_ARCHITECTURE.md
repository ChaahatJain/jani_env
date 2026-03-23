# DAgger Component Architecture - Verification Document

## Overview
The DAgger (Dataset Aggregation) algorithm has been decomposed into **modular, independently testable components** that follow the **Interface Segregation Principle** and **Dependency Inversion Principle**.

## Component Diagram

```
┌─────────────────────────────────────────────────────────┐
│                   DAgger Training Loop                   │
│                   (train.py / train.py)                  │
└──────────────────────┬──────────────────────────────────┘
                       │
         ┌─────────────┼─────────────────────┐
         │             │                     │
         ▼             ▼                     ▼
    ┌────────────┐ ┌──────────────┐ ┌──────────────────┐
    │  Sampler   │ │  Fault       │ │  Policy/Buffer   │
    │ Interface  │ │  Collector   │ │  Updater         │
    │            │ │  Interface   │ │  Interface       │
    └────────────┘ └──────────────┘ └──────────────────┘
         │             │                     │
         │             │                     │
    ┌─────────────────┴──────────────────────────────┐
    │                                                │
    ▼                                                ▼
┌──────────────────────┐            ┌──────────────────────┐
│ StandardTraceSampler │            │ OracleFaultCollector │
│ Implementation       │            │ Implementation       │
└──────────────────────┘            └──────────────────────┘
    │                                     │
    └──────────────────┬──────────────────┘
                       │
                ┌──────▼──────┐
                │  DAggerBuf  │
                │  (Training  │
                │   Dataset)  │
                └─────────────┘
```

## Component Breakdown

### 1. **Interfaces Layer** (`interfaces.py`)
Defines formal contracts for implementations using Abstract Base Classes:

#### PolicyInterface
```python
get_action(state, action_mask) -> action
```
- **Contract**: Any policy provider (NN, Shield, Heuristic) must implement this
- **Decoupling**: Separates algorithm from policy representation
- **Testability**: Can be mocked with dummy policies

#### TraceSamplerInterface
```python
sample_trace(env, policy, init_state_idx, max_steps) -> Dict[trace_data]
```
- **Contract**: Samples trajectories from environment using a policy
- **Returns**: Observations, actions, rewards, safety information
- **Testability**: Can test with mock environment

#### FaultCollectorInterface
```python
collect_faults(trace, oracle) -> List[faults]
```
- **Contract**: Extracts (state, corrected_action) pairs from traces
- **Input**: Safe/unsafe trajectory information
- **Output**: Training data for policy update
- **Testability**: Can test with mock traces

#### PolicyUpdaterInterface
```python
update_policy(policy, dataset) -> Dict[metrics]
```
- **Contract**: Updates policy weights using supervised dataset
- **Input**: Corrected action pairs
- **Output**: Training metrics
- **Testability**: Can test with synthetic datasets

### 2. **Implementations**

#### `sampler.py` - StandardTraceSampler
- **Role**: Implements TraceSamplerInterface
- **Responsibility**: Interacts with environment and policy
- **Key Methods**: 
  - Uses PolicyInterface for action selection (decoupled from policy type)
  - Handles episode termination and step counting
  - Returns structured trace data

#### `fault_collector.py` - OracleFaultCollector
- **Role**: Implements FaultCollectorInterface
- **Responsibility**: Analyzes traces and extracts faults
- **Key Methods**:
  - Iterates through trace (obs, action, mask) tuples
  - Calls oracle.evaluate_and_correct() for each step
  - Collects unsafe actions with corrected versions

#### `buffer.py` - DAggerBuffer & collect_trajectory
- **Role**: Manages training dataset
- **Responsibility**: Accumulates corrected actions
- **Key Methods**:
  - collect_trajectory(): Detailed trajectory collection with safety tracking
  - DAggerBuffer: Replay buffer for supervised learning

#### `policy.py` - Policy & Policy Network
- **Role**: Implements PolicyInterface via neural network
- **Responsibility**: Maps states to action logits
- **Key Methods**:
  - get_action(): Part of PolicyInterface contract
  - evaluate_policy_safety_on_state(): Safety evaluation

#### `policy_wrapper.py` - Policy Adapters
- **Role**: Wraps policies to implement interface
- **Responsibility**: Ensures compatibility with PolicyInterface
- **Use Case**: Adapting existing policies to work with sampler

#### `updater.py` - Policy Updater
- **Role**: Implements PolicyUpdaterInterface
- **Responsibility**: Supervised learning on corrected actions
- **Key Methods**:
  - update_policy(): Gradient descent on dataset

#### `training_scheduler.py` - Training Scheduler
- **Role**: Controls training dynamics
- **Responsibility**: Manages beta schedule and training loops
- **Key Methods**:
  - Schedules when to collect new data vs. update policy

#### `train.py` - Main Training Loop
- **Role**: Orchestrates all components
- **Responsibility**: Implements DAgger algorithm
- **Flow**:
  1. Sample trajectories using sampler + policy
  2. Collect faults using fault_collector + oracle
  3. Add to buffer
  4. Update policy using updater
  5. Repeat with new dataset aggregate

#### `ray_workers.py` - Distributed Training
- **Role**: Parallelization layer
- **Responsibility**: Enables multi-process policy evaluation
- **Use Case**: Scaling to many initial states

## Data Flow

```
Environment States
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Sampler (Using PolicyInterface)                          │
│ - Takes env + policy                                     │
│ - Returns: (obs, actions, action_masks, rewards, safety) │
└──────────────────┬───────────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ Traces (sequences)  │
         └────────┬────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Fault Collector (Using FaultCollectorInterface)         │
│ - Takes trace + oracle                                  │
│ - Extracts: (obs, action_mask, corrected_action)        │
│ - Returns: List of faults                               │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ DAggerBuffer        │
         │ (Training Dataset)  │
         └────────┬────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────┐
│ Updater (Using PolicyUpdaterInterface)                  │
│ - Takes dataset + policy                                │
│ - Performs supervised learning                          │
│ - Returns: Training metrics                             │
└──────────────────┬──────────────────────────────────────┘
                   │
                   ▼
         ┌─────────────────────┐
         │ Updated Policy      │
         │ (Implements         │
         │ PolicyInterface)    │
         └─────────────────────┘
             │
             └─────────┐ (Loop back to Sampler)
```

## Design Principles Applied

| Principle | Implementation | Benefit |
|-----------|----------------|---------|
| **Interface Segregation** | Separate interfaces for each component | Each component has focused responsibility |
| **Dependency Inversion** | Depend on abstractions, not concretions | Easy to swap implementations (NN, Shield, Heuristic) |
| **Single Responsibility** | Sampler samples, Collector collects, Updater updates | Clear testing boundaries |
| **Open/Closed** | Closed for modification, open for extension | Add new policy types without changing sampler |
| **Composition** | Train.py orchestrates interfaces | Flexible algorithm variants |

## Testability Features

✅ **Mockable Policies**: PolicyInterface allows any policy type
✅ **Mockable Environments**: TraceSampler works with any env
✅ **Mockable Oracles**: FaultCollector decoupled from oracle
✅ **Deterministic Traces**: Can replay and verify safety
✅ **Isolated Components**: Test each component independently
✅ **Verifiable Data Flow**: Each step produces testable outputs

## Next Steps for Verification

1. **Unit Tests**: Test each interface implementation independently
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Run full DAgger pipeline
4. **Safety Verification**: Verify oracle correctness
5. **Performance Benchmarks**: Compare against baselines

