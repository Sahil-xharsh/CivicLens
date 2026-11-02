# CivicLens label guide

This guide defines the labels for the CivicLens complaint pilot. It is a research labeling reference, not an operational emergency service protocol.

## Categories

Choose the main civic service responsible for the reporeted problem. Use the specific category when it fits - use `other` when the issue does not fit one of the listed services.

| Label | Use for |
| --- | --- |
| `roads` | Road surfaces, potholes, road trenches, and other roadway damage. |
| `water` | Drinking-water supply, pressure, pipes, and leaks. |
| `sanitation` | Garbage collection, public toilets, and routine waste handling. |
| `electricity` | Power supply, transformers, and electrical cables or wires. |
| `drainage` | Drains, sewers, waterlogging, and flooding caused by drainage problems. |
| `streetlights` | Public street lamps and their poles, unless a separate hazard is the main issue. |
| `other` | Civic issues outside the categories above, including hazardous-material spills that need specialist handling. |

When a complaint mentions more than one issue, label the primary reported service problem. Record immediate risk separately with `is_emergency`; do not change the category just because a hazard is dangerous.

## Urgency

Urgency describes how quickly the service problem needs attention, apart from whether an emergency hazard is present.

| Label | Rule of thumb |
| --- | --- |
| `low` | Routine repair or service request without significant disruption. |
| `medium` | A localized problem causing inconvenience or a minor service interruption. |
| `high` | A worsening or widespread disruption or a hazard needing prompt attention, without a clearly immediate threat to people. |
| `critical` | An active, immediate threat to people or occupied property or a severe incident already causing imminent harm. |

Use the facts stated in the complaint. Do not infer injury, exposure or imminent danger that the text does not support.

## Emergency risk

Set `is_emergency` to `true` when the text describes a current or imminent serious risk to people, such as an accessible live wire, an unguarded dangerous excavation, a hazardous spill near people, or people trapped by rising floodwater. Set it to `false` for routine faults and service interruptions without such a risk.

Emergency risk is a separate binary label, not another urgency level. A complaint can be `high` and emergency-risk `true` when a serious hazard needs immediate human assessment but the text does not establish an active, immediate threat. Use `critical` when the text does establish that immediate threat. Ambiguous potential hazards should be flagged for human review rather than labeled safe.

## Pilot decisions

- A hazardous-waste spill near children is `other`, because specialist hazardous-material handling is the primary service need; it is `critical` and emergency-risk `true` because people are nearby.
- A cable explicitly described as live and fallen across a public footpath is `electricity`, `critical`, and emergency-risk `true`, even if sparking is not mentioned.
- Intermittent transformer sparks are `electricity`, `high`, and emergency-risk `true`: the text describes a serious electrical hazard, but not a clearly immediate threat at the time of reporting.
- For the same issue described with stronger evidence of immediate danger, such as a live wire sparking beside pedestrians, use `critical` and emergency-risk `true`.

## Dataset values

Categories: `roads`, `water`, `sanitation`, `electricity`, `drainage`, `streetlights`, `other`.

Urgency: `low`, `medium`, `high`, `critical`.

Languages: `en`, `hi`, `hinglish`, `mixed`.

The pilot uses `synthetic` as its label source and `train` as its split. Synthetic pilot examples are for checking the data format and label interpretation; they are not evaluation data and must not support performance claims.
