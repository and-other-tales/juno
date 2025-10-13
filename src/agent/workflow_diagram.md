<!-- Copyright © 2025 PI & Other Tales Inc.. All Rights Reserved. -->
# Othertales Juno Workflow Diagram

The following Mermaid diagram illustrates the self-improving workflow for Othertales Juno, showing how the system evaluates its performance, identifies resource constraints, and deploys improved code and resources.

```mermaid
flowchart TD
    A[Task Received] --> B[Set Deadline & Task Size]
    B --> C[Check Resource Sufficiency]
    C -- "Resources Sufficient" --> D[Execute Task]
    C -- "Resources Insufficient" --> N[Flag Resource Constraint]
    N --> D

    D --> E[Research Team]
    D --> F[Writing Team]
    E --> G[Research Result]
    F --> H[Writing Result]
    
    G --> I[Supervisor Feedback]
    H --> I
    
    I --> J{Quality Check}
    J -- "Quality OK" --> K{Deadline Check}
    J -- "Low Quality" --> L[Quality Count++]
    L --> M{3 Low Quality?}
    M -- "No" --> K
    M -- "Yes" --> Q[Create Improvement Request]
    
    K -- "Deadline Met" --> P[Complete Task]
    K -- "Deadline Missed" --> O[Missed Deadline Count++]
    O --> R{Missed > 2?}
    R -- "No" --> P
    R -- "Yes" --> Q
    
    Q --> S[Route to Juno Team]
    
    S --> T[Juno Supervisor]
    T --> U[Evaluator Agent]
    U --> V[Performance Analysis]
    V --> W{Issue Type?}
    
    W -- "Code Issues" --> X[Code Agent]
    W -- "Resource Constraints" --> Y[Resource Scaling]
    
    X --> Z[Implement Code Fixes]
    Z --> AA[Test Fixes]
    AA --> AB[Deploy Code Changes]
    
    Y --> AC[Add Resources to Team]
    AC --> AD[Monitor New Resources]
    AD --> AE[Resource Performance Report]
    
    AB --> AF[Verify Improvements]
    AE --> AF
    
    AF --> AG[Next Cycle]
    P --> AG
    
    %% Random workload increase logic
    AG --> AH{Random Increase?}
    AH -- "Yes" --> AI[Increase Task Size]
    AH -- "No" --> AJ[Standard Task Size]
    AI --> AK[Set Tighter Deadline]
    AJ --> AL[Set Standard Deadline]
    AK --> A
    AL --> A
    
    %% Styling
    classDef process fill:#f9f,stroke:#333,stroke-width:2px;
    classDef decision fill:#bbf,stroke:#333,stroke-width:2px;
    classDef result fill:#bfb,stroke:#333,stroke-width:2px;
    
    class A,B,D,E,F,I,L,O,Q,S,T,U,V,X,Y,Z,AA,AB,AC,AD,AE,AI,AJ,AK,AL process;
    class C,J,K,M,R,W,AH decision;
    class G,H,P,AF,AG result;
```

## Key Workflow Components

1. **Task Management**
   - Task received with deadline
   - Dynamic workload adjustments
   - Resource sufficiency checks

2. **Execution**
   - Research and writing teams process tasks
   - Supervisor evaluates outputs
   - Quality and deadline tracking

3. **Issue Identification**
   - Consecutive low quality triggers improvement
   - Missed deadlines trigger resource analysis
   - Performance metrics signal resource constraints

4. **Juno Team Response**
   - Evaluator analyzes performance data
   - Code agent implements fixes for quality issues
   - Resource scaling when deadlines missed
   
5. **Resource Management**
   - Add resources to constrained teams
   - Monitor performance of new resources
   - Adjust resource allocation based on metrics

6. **Cycle Restart**
   - Random workload increase (30% probability)
   - Task size can increase up to 2x
   - Deadlines adjusted based on task size