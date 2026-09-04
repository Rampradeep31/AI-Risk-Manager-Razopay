# Problem Statement

At the moment an order is placed, we want to estimate the probability that the order will
eventually be returned by the customer — a single, precisely-scoped class of loss: **return risk**,
not fraud, not chargebacks, not delivery failure. The output is a **risk score in [0, 1]**, computed
from information available at order time only (customer history, order composition, payment
method, delivery promise, device/context signals). This score is consumed by the fulfillment and
customer-ops team as an **advisory signal**, not an automated blocker: orders above the chosen
threshold are routed to a lightweight review queue (e.g. a size/fit confirmation nudge, a
restricted payment method, or additional packaging for fragile/return-prone categories), while
everything else ships through the normal path with zero added friction. Getting this wrong in
either direction has an asymmetric, quantifiable cost. A **false positive** — flagging a legitimate,
low-risk order — adds friction to a real customer (a support touch, a delayed shipment, erosion of
trust) even though the order was never going to be returned. A **false negative** — missing an
order that will actually be returned — costs the business the full return-handling economics
(reverse shipping, restocking/inspection labor, and lost margin on an order that generated revenue
but no retained sale) with no chance to intervene. Because these costs are not symmetric and the
outcome class is inherently imbalanced (most orders are not returned), the model must be evaluated
and thresholded on precision/recall and an explicit expected-cost basis rather than plain accuracy,
and every flagged decision must carry a mathematically grounded explanation (via SHAP) so ops staff
can trust and act on it rather than treat the score as an opaque black box.
