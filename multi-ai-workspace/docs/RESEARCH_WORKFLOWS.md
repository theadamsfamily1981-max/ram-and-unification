# Research Workflows - Advanced Multi-AI Research

**Version:** 1.2.0
**Focus:** Conversational AI with multi-AI orchestration for advanced research

---

## Overview

The **Research Avatar** is a conversational AI assistant that conducts advanced research by orchestrating multiple AI specialists. It engages in natural dialogue, intelligently delegates to online AIs for specialized tasks, and synthesizes findings into coherent narratives.

### Key Capabilities

✅ **Conversational Interface** - Natural, flowing research conversations
✅ **Multi-AI Orchestration** - Coordinates Claude, Nova, Pulse, and Ara
✅ **Intelligent Delegation** - Routes queries to appropriate specialists
✅ **Synthesis Engine** - Combines findings into unified narratives
✅ **Control Knobs** - User-configurable research behavior
✅ **Privacy-First** - Sensitive data stays offline

---

## Architecture

```
┌─────────────────────────────────────────────┐
│        Research Avatar (Conversational)      │
│           Mistral 7B / Mixtral 8x7B          │
│                                              │
│  • Natural conversation interface            │
│  • Understands research questions            │
│  • Plans research strategy                   │
│  • Orchestrates multiple AIs                 │
│  • Synthesizes findings                      │
└───────────────────┬──────────────────────────┘
                    │
     ┌──────────────┴──────────────┐
     │   Research Orchestration     │
     └──────────────┬──────────────┘
                    │
    ┌───────────────┼───────────────┬────────────┐
    │               │               │            │
    ▼               ▼               ▼            ▼
┌────────┐     ┌────────┐     ┌────────┐   ┌────────┐
│ Claude │     │  Nova  │     │ Pulse  │   │  Ara   │
│Technical│    │General │     │ Large  │   │Unique  │
│ Expert │     │Purpose │     │Context │   │Views   │
└────────┘     └────────┘     └────────┘   └────────┘
     │              │               │            │
     └──────────────┴───────────────┴────────────┘
                    │
              ┌─────▼─────┐
              │ Synthesis │
              │  Engine   │
              └───────────┘
```

---

## Research Modes

### 1. Literature Review

**Use Case:** Comprehensive research from multiple perspectives

**AIs Consulted:** Claude + Nova + Pulse (parallel)

**Strategy:** Gather diverse viewpoints simultaneously

**Example:**
```
User: "Research the state of quantum computing"

Avatar: "Great topic! I'll conduct a literature review using:
         🔵 Claude - Technical/academic analysis
         🟢 Nova - Practical applications and trends
         🔴 Pulse - Comprehensive industry overview

         [Consulting 3 AIs in parallel...]

         **Findings:**

         **Technical State (Claude):**
         Quantum computing has achieved quantum supremacy in specific
         tasks. Current systems use 50-100 qubits but face decoherence...

         **Applications (Nova):**
         Promising areas include cryptography, drug discovery, and
         optimization. IBM and Google lead commercial development...

         **Industry Overview (Pulse):**
         $30B invested 2020-2024. Major players: IBM, Google, IonQ...

         **My Synthesis:**
         Quantum computing is transitioning from research to early
         commercial applications. While true quantum advantage remains
         limited, the field shows promising progress in..."
```

**Control Knobs:**
```yaml
research_depth: standard  # Or: quick, deep, exhaustive
source_diversity: multi   # Using 3 AIs
```

---

### 2. Deep Dive

**Use Case:** Expert-level analysis on focused topics

**AIs Consulted:** Pulse (planning) → Claude (execution)

**Strategy:** Plan first, then deep technical analysis

**Example:**
```
User: "Deep dive into React hooks performance"

Avatar: "This calls for a deep dive. My approach:

         Step 1: Pulse will create a comprehensive analysis plan
         Step 2: Claude will execute deep technical investigation

         [Pulse planning...]

         Pulse created this research plan:
         1. How hooks work under the hood
         2. Performance comparison with class components
         3. Common performance pitfalls
         4. Optimization strategies
         5. Benchmarking methodology

         [Claude executing deep analysis...]

         **Technical Analysis (Claude):**
         [Detailed technical findings with code examples...]

         Want me to drill deeper into any specific aspect?"
```

**Control Knobs:**
```yaml
research_depth: deep
verbosity: detailed
response_time: thorough
```

---

### 3. Comparative Analysis

**Use Case:** Compare options, evaluate trade-offs, get diverse perspectives

**AIs Consulted:** All 4 AIs (parallel)

**Strategy:** Maximum perspective diversity

**Example:**
```
User: "Compare PostgreSQL vs MongoDB vs Cassandra"

Avatar: "Perfect for comparative analysis! I'll consult all 4 AIs:

         🔵 Claude - Technical architecture and performance
         🟢 Nova - Developer experience and ecosystem
         🔴 Pulse - Enterprise scale and long-term considerations
         🟣 Ara - Alternative perspectives and edge cases

         [Consulting 4 AIs in parallel...]

         **Technical Performance (Claude):**
         PostgreSQL: ACID compliance, complex queries, mature optimizer
         MongoDB: Flexible schema, horizontal scaling, document model
         Cassandra: Linear scalability, high write throughput, eventual consistency

         **Developer Experience (Nova):**
         PostgreSQL: SQL familiarity, robust tooling, extensive resources
         MongoDB: Intuitive for JSON-like data, rapid prototyping
         Cassandra: Steeper learning curve, but powerful for distributed systems

         **Enterprise Considerations (Pulse):**
         [Analysis of costs, support, hiring, long-term maintenance...]

         **Alternative View (Ara):**
         "Consider NewSQL options like CockroachDB for distributed SQL,
          or evaluate if you really need NoSQL at all..."

         **My Synthesis:**
         Choose based on your specific needs:
         - Strong consistency + complex queries → PostgreSQL
         - Flexible schema + rapid iteration → MongoDB
         - Massive scale + high writes → Cassandra

         Want me to explore any database in more detail?"
```

**Control Knobs:**
```yaml
source_diversity: all
synthesis_style: comparison
```

---

### 4. Quick Synthesis

**Use Case:** Fast research for time-sensitive questions

**AIs Consulted:** Nova OR Pulse (competitive)

**Strategy:** First to respond wins

**Example:**
```
User: "Quick: What's the difference between REST and GraphQL?"

Avatar: "Quick synthesis mode! Racing Nova and Pulse...

         [Competitive query - first response wins...]

         Nova responded first (0.8s):

         **REST:**
         - Multiple endpoints (/users, /posts, etc.)
         - Over-fetching/under-fetching common
         - Easier caching
         - Simpler to understand

         **GraphQL:**
         - Single endpoint
         - Request exactly what you need
         - More complex caching
         - Steeper learning curve

         Use REST for simple APIs, GraphQL for complex data relationships.

         Need more detail? I can do a deep dive!"
```

**Control Knobs:**
```yaml
research_depth: quick
response_time: fast
verbosity: concise
```

---

### 5. Multi-Stage Research

**Use Case:** Complex research requiring multiple steps

**AIs Consulted:** Sequential workflow across multiple AIs

**Strategy:** Step-by-step with synthesis

**Example:**
```
User: "Research and recommend a tech stack for a social media startup"

Avatar: "This needs multi-stage research. Here's my plan:

         **Stage 1: Planning** (Pulse)
         Creating comprehensive research framework...

         **Stage 2: Data Gathering** (Claude + Nova + Pulse, parallel)
         Consulting all three main AIs for information...

         **Stage 3: Synthesis** (Me - Avatar)
         Combining findings into recommendations...

         **Stage 4: Critique** (Ara)
         Getting alternative perspective and challenges...

         **Stage 5: Final Recommendation** (Me - Avatar)
         Integrating all insights...

         [Executing multi-stage workflow...]

         **Stage 1: Research Plan (Pulse):**
         Key areas to investigate:
         1. Frontend framework
         2. Backend architecture
         3. Database selection
         4. Infrastructure/hosting
         5. Real-time features
         6. Scalability considerations

         **Stage 2: Information Gathering:**

         Claude's technical analysis:
         [Detailed technical stack recommendations...]

         Nova's practical insights:
         [Developer experience, hiring, learning curves...]

         Pulse's comprehensive overview:
         [Industry trends, cost analysis, long-term viability...]

         **Stage 3: My Synthesis:**
         Recommended stack:
         - Frontend: Next.js (React) - SSR, great DX, huge ecosystem
         - Backend: Node.js + GraphQL - unified language, efficient data
         - Database: PostgreSQL (primary) + Redis (cache) - proven scale
         - Infrastructure: Vercel (frontend) + AWS (backend) - easy deploy
         - Real-time: WebSockets via Socket.io - simple, reliable

         **Stage 4: Ara's Critique:**
         "Consider Remix instead of Next.js for better data loading..."
         "Evaluate whether GraphQL complexity is worth it for v1..."
         "PostgreSQL is safe, but MongoDB might accelerate early iteration..."

         **Stage 5: Final Recommendation:**

         For a social media startup, I recommend:

         **MVP (Months 0-6):**
         Next.js + Node.js + PostgreSQL + Vercel + AWS
         Rationale: Proven, fast development, easy hiring

         **Scale (Months 6-18):**
         Add Redis caching, consider GraphQL, optimize DB queries

         **Alternatives to Consider:**
         - Remix (if you want cutting-edge React)
         - MongoDB (if schema flexibility is critical)
         - Supabase (if you want Firebase-like experience)

         Want me to deep dive into any part of this stack?"
```

**Control Knobs:**
```yaml
research_depth: exhaustive
verbosity: detailed
source_diversity: all
synthesis_style: narrative
```

---

## Control Knobs

### Configuration

Control knobs are defined in `config/research_avatar.yaml`:

```yaml
control_knobs:
  research_depth:
    type: enum
    options: [quick, standard, deep, exhaustive]
    default: standard

  verbosity:
    type: enum
    options: [concise, balanced, detailed, verbose]
    default: balanced

  source_diversity:
    type: enum
    options: [single, dual, multi, all]
    default: multi

  privacy_mode:
    type: enum
    options: [offline_only, hybrid, online_preferred]
    default: hybrid

  response_time:
    type: enum
    options: [instant, fast, balanced, thorough]
    default: balanced

  synthesis_style:
    type: enum
    options: [summary, comparison, narrative, academic]
    default: narrative

  confidence_threshold:
    type: float
    range: [0.0, 1.0]
    default: 0.7
```

### Usage

#### Via API

```bash
curl -X POST http://localhost:8000/api/send \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Research blockchain scalability",
    "context": {
      "pack": "research-assistant",
      "control_knobs": {
        "research_depth": "deep",
        "source_diversity": "all",
        "verbosity": "detailed",
        "synthesis_style": "academic"
      }
    }
  }'
```

#### Via Tags

```bash
# Quick research
"#quick #offline Research REST APIs"

# Deep research with all AIs
"#deep #multiverse Analyze ML frameworks"

# Private research (offline only)
"#private #offline Research medical data privacy"
```

### Control Knob Matrix

| Setting | Quick | Standard | Deep | Exhaustive |
|---------|-------|----------|------|------------|
| **AIs Used** | 1 | 2-3 | 3-4 | All + multi-stage |
| **Strategy** | Competitive | Parallel | Sequential | Multi-stage workflow |
| **Time** | <2s | 3-5s | 5-10s | 10-30s |
| **Depth** | Surface | Good | Comprehensive | Exhaustive |
| **Cost** | $ | $$ | $$$ | $$$$ |

---

## Example Workflows

### Workflow 1: Academic Paper Research

**Scenario:** Researching for thesis on distributed systems

```
User: "I'm writing my thesis on distributed consensus algorithms.
       I need comprehensive research on Paxos, Raft, and new approaches."

Avatar: "Excellent thesis topic! This needs deep, academic research.
         Let me ask a few clarifying questions:

         1. What's your specific angle? (Theory, implementation, comparison?)
         2. Do you need historical context or just current state?
         3. Are you implementing something or purely theoretical?

User: "Comparative analysis of Paxos vs Raft vs newer algorithms.
       I'm implementing a toy version to understand trade-offs."

Avatar: "Perfect! This calls for **exhaustive research mode**.

         My plan:
         1. Pulse: Create comprehensive research framework
         2. Claude: Deep technical analysis (best for algorithms)
         3. Nova: Practical implementation insights
         4. Pulse: Large-scale deployment considerations
         5. Ara: Critique assumptions and suggest novel approaches
         6. Me: Synthesize into academic-quality analysis

         Control knobs:
         - research_depth: exhaustive
         - verbosity: detailed
         - source_diversity: all
         - synthesis_style: academic

         [Executing multi-stage research...]

         **Stage 1: Research Framework (Pulse):**

         Key areas to investigate:
         1. Theoretical foundations
            - FLP impossibility theorem
            - CAP theorem implications
         2. Paxos family
            - Basic Paxos
            - Multi-Paxos
            - Fast Paxos
         3. Raft
            - Leader election
            - Log replication
            - Safety properties
         4. Modern alternatives
            - EPaxos
            - Flexible Paxos
            - Byzantine fault tolerance (PBFT, HotStuff)
         5. Implementation trade-offs
         6. Performance characteristics

         **Stage 2: Technical Analysis (Claude):**

         [Claude provides detailed algorithmic analysis with pseudocode...]

         **Paxos:**
         - Theoretical elegance but complex implementation
         - Proof of correctness established by Lamport (1998)
         - Challenge: Multiple rounds required for consensus

         **Raft:**
         - Designed explicitly for understandability
         - Strong leader approach simplifies reasoning
         - Log-structured, easier to implement
         - Performance comparable to Paxos

         **Modern Approaches:**

         EPaxos (Egalitarian Paxos):
         - Leaderless, reduces latency
         - Handles commutative operations efficiently

         Flexible Paxos:
         - Relaxes Paxos quorum requirements
         - Better performance in asymmetric deployments

         **Stage 3: Implementation Insights (Nova):**

         [Nova provides practical implementation advice...]

         For your toy implementation:
         - Start with Raft (clearer mental model)
         - Use gRPC for network communication
         - Implement state machines separately
         - Focus on leader election first

         Resources:
         - Raft visualization: raft.github.io
         - Paper: "In Search of an Understandable Consensus Algorithm"
         - Implementations: etcd (Go), Consul (Go)

         **Stage 4: Deployment Considerations (Pulse):**

         [Pulse analyzes production usage...]

         Real-world usage:
         - Raft: etcd (Kubernetes), Consul, CockroachDB
         - Paxos: Google Chubby, Spanner
         - Byzantine: Tendermint (Cosmos), HotStuff (LibraBFT)

         Performance at scale:
         - Raft: 10K ops/sec (5 nodes)
         - EPaxos: 2x throughput in low-contention
         - Network latency dominant factor

         **Stage 5: Critical Perspective (Ara):**

         [Ara challenges assumptions...]

         Alternative angle: "Why are we still using consensus algorithms
         from the 80s and 90s?"

         - CRDTs (Conflict-free Replicated Data Types) avoid consensus entirely
         - Operational transforms for collaborative editing
         - Hybrid approaches: consensus only for critical operations

         For your thesis, consider:
         - Is strong consistency always needed?
         - Eventual consistency trade-offs
         - Domain-specific optimizations

         **Stage 6: Academic Synthesis (Avatar):**

         **Distributed Consensus: A Comparative Analysis**

         **Abstract:**
         Distributed consensus remains a fundamental challenge in building
         reliable distributed systems. This analysis examines three major
         approaches: Paxos, Raft, and modern alternatives, evaluating their
         theoretical foundations, implementation complexity, and practical
         performance characteristics.

         **1. Introduction**

         The consensus problem, formalized by Fischer, Lynch, and Paterson
         (FLP, 1985), establishes that no deterministic algorithm can guarantee
         consensus in an asynchronous system with even one faulty process.
         Despite this impossibility result, practical consensus algorithms
         achieve reliability through various mechanisms...

         **2. Paxos Family**

         **2.1 Theoretical Foundation**
         Paxos, introduced by Lamport (1998), provides provably correct
         consensus under the assumption of asynchronous networks with
         non-Byzantine failures...

         [Continues with academic-style analysis...]

         **3. Raft**

         Motivated by Paxos's reputation for difficulty, Ongaro and
         Ousterhout designed Raft explicitly for understandability...

         **4. Modern Alternatives**

         **4.1 EPaxos**
         [Analysis of leaderless approaches...]

         **4.2 Flexible Paxos**
         [Quorum flexibility discussion...]

         **4.3 Byzantine Fault Tolerance**
         [PBFT, HotStuff, practical BFT...]

         **5. Comparative Analysis**

         | Algorithm | Complexity | Performance | Use Case |
         |-----------|------------|-------------|----------|
         | Paxos | High | Moderate | Theoretical foundation |
         | Raft | Medium | Moderate | Production systems |
         | EPaxos | High | High | Low-latency geo-distributed |
         | Flexible Paxos | Medium | High | Asymmetric deployments |

         **6. Implementation Considerations**

         For your thesis implementation:
         1. Begin with Raft for clear understanding
         2. Implement state machine separately
         3. Focus on correctness before performance
         4. Use simulation testing (Jepsen-style)

         **7. Future Directions**

         Emerging research areas:
         - Application-specific consensus
         - Machine learning-guided leader election
         - Quantum-resistant consensus
         - Consensus-free architectures (CRDTs)

         **8. Conclusion**

         While Paxos provides theoretical elegance, Raft's understandability
         has driven widespread adoption. Modern alternatives like EPaxos show
         promise for specific use cases, but the fundamental trade-offs
         established by FLP remain relevant...

         **References:**
         [Academic citation list...]

         ---

         This should give you a strong foundation for your thesis!

         Next steps:
         - Want me to dive deeper into any algorithm?
         - Need help with implementation architecture?
         - Should I research testing strategies (Jepsen)?

         I can also generate code examples or explain specific proofs."
```

**Result:** Comprehensive, academic-quality research with multiple perspectives, suitable for thesis work.

---

### Workflow 2: Business Decision Research

**Scenario:** Startup founder choosing technology stack

```
User: "We're building a fintech app. Help me choose between
       microservices vs monolith, and AWS vs GCP vs Azure."

Avatar: "Important decisions! Let me get some context:

         1. Team size and experience level?
         2. Expected scale in next 12 months?
         3. Regulatory requirements (PCI-DSS, etc.)?
         4. Budget constraints?

User: "5-person team, mostly backend. Expect 10K users in 6 months,
       maybe 100K in 12 months. Need PCI compliance. $10K/month cloud budget."

Avatar: "Thanks! This needs **comparative analysis** with practical focus.

         I'll consult:
         - Claude: Technical architecture assessment
         - Nova: Team dynamics and practical considerations
         - Pulse: Cost analysis and cloud provider comparison
         - Ara: Challenge assumptions, suggest alternatives

         [Consulting 4 AIs...]

         **Technical Assessment (Claude):**

         **Monolith vs Microservices:**
         For 5-person team + 100K users → **Monolith**

         Rationale:
         - Microservices overhead not justified at this scale
         - Easier debugging and deployment
         - Can always break apart later (strangler pattern)
         - PCI compliance easier with fewer moving parts

         Recommended architecture:
         - Modular monolith (clear boundaries)
         - Separate payment service (PCI scope reduction)
         - API gateway for future scalability

         **Team Dynamics (Nova):**

         With 5 backend developers:
         - Monolith keeps team velocity high
         - Microservices would add operational burden
         - Focus energy on product, not infrastructure

         However, prepare for transition:
         - Use domain-driven design
         - Clear module boundaries
         - CI/CD from day one

         **Cloud Provider Comparison (Pulse):**

         **AWS:**
         Pros: Mature, most services, best for compliance (PCI DSS Level 1)
         Cons: Complex pricing, steeper learning curve
         Est. cost: $8K-12K/month

         **GCP:**
         Pros: Simpler, better Kubernetes, ML tools
         Cons: Smaller compliance footprint
         Est. cost: $6K-10K/month

         **Azure:**
         Pros: Good enterprise features, hybrid cloud
         Cons: Less fintech-focused than AWS
         Est. cost: $7K-11K/month

         For fintech + PCI compliance → **AWS**

         Recommended AWS stack:
         - ECS Fargate (containers without K8s complexity)
         - RDS PostgreSQL (managed, compliant)
         - Elasticache Redis (sessions, caching)
         - CloudFront + WAF (security)
         - Stripe for payment processing (outsource PCI)

         Total: ~$9K/month

         **Alternative Perspective (Ara):**

         "Everyone says monolith first, but consider:

         - Serverless (Lambda + API Gateway) for even simpler ops
         - Railway or Render for faster deployment than AWS
         - Supabase (PostgreSQL + auth + APIs) as backend

         For fintech, challenge the assumption that you need AWS.
         Vercel + Supabase + Stripe might get you to 100K users for $2K/month.

         Only go AWS if you have AWS expertise or specific compliance needs."

         **My Synthesis:**

         **Recommendation: Modular Monolith on AWS**

         Architecture:
         ```
         [Frontend] → [API Gateway] → [Monolith] → [PostgreSQL]
                            ↓
                     [Separate Payment Service] → [Stripe]
         ```

         **Why monolith:**
         - Team size (5) can't support microservices overhead
         - Scale target (100K users) doesn't require microservices
         - Faster development velocity
         - Easier PCI compliance

         **Why AWS:**
         - Best for financial compliance (PCI DSS Level 1)
         - Mature security services
         - Within budget ($9K/month)

         **Implementation Plan:**

         **Phase 1 (Months 0-3):**
         - Modular monolith on ECS Fargate
         - RDS PostgreSQL
         - Stripe for payments (offload PCI)
         - CloudFront + WAF

         **Phase 2 (Months 3-6):**
         - Add Redis caching
         - Optimize database queries
         - Implement monitoring (CloudWatch)

         **Phase 3 (Months 6-12):**
         - Extract payment service if needed
         - Consider read replicas
         - Plan microservices transition if scaling beyond 100K

         **Alternative if budget is tight:**
         Ara's suggestion of Vercel + Supabase + Stripe is valid for
         MVP. You can always migrate to AWS later if needed.

         Trade-off: $2K/month vs $9K/month, but less control and
         potential migration pain later.

         **My final recommendation:**
         - MVP (first 3 months): Vercel + Supabase + Stripe
         - Production (after product-market fit): AWS + Modular Monolith

         This lets you validate quickly, then scale properly.

         Want me to dive deeper into any aspect?"
```

**Result:** Practical, business-focused recommendation with clear reasoning and implementation plan.

---

## Usage Guide

### Starting a Research Session

```bash
# Start research avatar
curl -X POST http://localhost:8000/api/send \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "I want to research [topic]",
    "context": {
      "pack": "research-assistant"
    }
  }'
```

### Configuring Research Behavior

```bash
# Deep research with all AIs
curl -X POST http://localhost:8000/api/send \
  -d '{
    "prompt": "Research [topic]",
    "context": {
      "pack": "research-assistant",
      "control_knobs": {
        "research_depth": "exhaustive",
        "source_diversity": "all",
        "verbosity": "detailed"
      }
    }
  }'
```

### Using Tags for Quick Configuration

```bash
# Quick research
"#quick Research machine learning"

# Deep multi-AI research
"#deep #multiverse Analyze blockchain scalability"

# Offline private research
"#offline #private Research medical data handling"
```

---

## Best Practices

### 1. Start with Clarifying Questions

Let the avatar ask questions to understand your research needs:

```
User: "Research cloud providers"

Avatar: "I'd love to help! To give you the best research:
         - What's your use case? (Web app, ML, data processing?)
         - Team size and expertise?
         - Budget range?
         - Geographic requirements?"
```

### 2. Use Appropriate Research Depth

Match research depth to your needs:

- **Quick:** Time-sensitive, surface-level
- **Standard:** Balanced research for most cases
- **Deep:** Important decisions, comprehensive analysis
- **Exhaustive:** Critical research, thesis-level

### 3. Leverage Multiple Perspectives

For important decisions, use `source_diversity: all`:

```yaml
control_knobs:
  source_diversity: all  # Get all 4 AI perspectives
```

### 4. Iterate and Refine

Research is conversational - ask follow-up questions:

```
Avatar: "[Provides research synthesis]"

User: "Can you dive deeper into the security aspects?"

Avatar: "Absolutely! Let me get Claude to do a security deep-dive..."
```

### 5. Privacy First

Mark sensitive research as private:

```bash
"#private #offline Research employee retention strategies"
```

---

## Monitoring and Feedback

### Track Research Quality

```bash
# Get research session stats
curl http://localhost:8000/api/research/stats
```

### Provide Feedback

```bash
# Rate research quality
curl -X POST http://localhost:8000/api/research/feedback \
  -d '{
    "session_id": "...",
    "rating": 5,
    "comments": "Excellent synthesis!"
  }'
```

---

## Cost Optimization

### Estimated Costs per Research Mode

| Mode | AIs Used | Avg Tokens | Cost/Query |
|------|----------|------------|------------|
| Quick | 1 | 1K | $0.001 |
| Standard | 2-3 | 5K | $0.015 |
| Deep | 3-4 | 15K | $0.045 |
| Exhaustive | All + multi-stage | 40K | $0.120 |

### Cost-Saving Strategies

1. **Use offline avatar for simple queries** → $0.00
2. **Start with quick mode** → Upgrade to deep if needed
3. **Cache research results** → Avoid duplicate queries
4. **Batch related questions** → Single research session

---

## Troubleshooting

### Issue: Research takes too long

**Solution:** Adjust response time preference:

```yaml
control_knobs:
  response_time: fast  # Use competitive strategy
```

### Issue: Not enough depth

**Solution:** Increase research depth:

```yaml
control_knobs:
  research_depth: deep  # Or exhaustive
  verbosity: detailed
```

### Issue: Too many AI perspectives (overwhelming)

**Solution:** Reduce source diversity:

```yaml
control_knobs:
  source_diversity: dual  # Just 2 AIs
  synthesis_style: summary  # Concise synthesis
```

---

## Summary

The Research Avatar provides:

✅ **Conversational research** - Natural dialogue, not just Q&A
✅ **Multi-AI orchestration** - Leverage multiple specialists
✅ **Intelligent synthesis** - Unified narratives, not raw responses
✅ **Configurable behavior** - Control knobs for every need
✅ **Privacy protection** - Sensitive data stays offline
✅ **Cost optimization** - Smart delegation reduces API costs

**Perfect for:**
- Academic research
- Business decisions
- Technical evaluations
- Comparative analysis
- Literature reviews
- Deep dives on complex topics

---

**Get started:**

```bash
curl -X POST http://localhost:8000/api/send \
  -d '{"prompt": "I want to research [your topic]", "context": {"pack": "research-assistant"}}'
```

The avatar will guide you through the research process!
