import joblib
import pandas as pd
import numpy as np
from scipy.optimize import root_scalar


# ============================================================================
# MODEL LOADING AND INITIALIZATION
# ============================================================================
# Benefits of lazy loading approach:
# - Faster API startup time (models only loaded when first needed)
# - Memory efficient for services that may not use all endpoints
# Trade-off: First request has slight latency while loading models

# Load all models and limits once
_MODELS_PATH = "api/all_models.pkl"
_LIMITS_PATH = "data/asset_limits.csv"

_all_models = None
_limits = None

def _load_models_and_limits():
    global _all_models, _limits
    if _all_models is None:
        _all_models = joblib.load(_MODELS_PATH)
    if _limits is None:
        _limits = pd.read_csv(_LIMITS_PATH, dtype={"asset": str})
        _limits["asset"] = _limits["asset"].astype(str)
        _limits.set_index("asset", inplace=True)


# ============================================================================
# CORE PREDICTION FUNCTION: R_clean(HP)
# ============================================================================
# PRESENTATION KEY POINT:
# This is the heart of our modeling approach - a hybrid prediction system that
# combines data-driven isotonic regression with physics-based linear extrapolation.
#
# BENEFITS OF ISOTONIC REGRESSION:
# 1. Non-parametric: Makes no assumptions about functional form
# 2. Monotonic guarantee: Ensures restriction increases with HP (physically correct)
# 3. Flexible: Captures non-linear relationships in historical data
# 4. Robust: Less sensitive to outliers than polynomial regression
#
# TRADE-OFFS vs OTHER APPROACHES:
# vs Linear Regression:
#   + Better: Captures non-linear patterns in real data
#   - Worse: Cannot extrapolate beyond training data
# vs Polynomial Regression:
#   + Better: No risk of unrealistic oscillations, guaranteed monotonicity
#   - Worse: Less smooth in some regions
# vs Neural Networks:
#   + Better: Interpretable, guaranteed monotonicity, works with small datasets
#   - Worse: May not capture extremely complex patterns
# 
# OUR SOLUTION: Hybrid approach uses isotonic for fitted range (data-driven)
# and linear extrapolation beyond (physics-based conservative estimates)

def _predict_restriction_from_hp(asset_type, hp):
    """
    Given an asset_type and HP value, predict restriction using isotonic model + extrapolation.
    This is R_clean(HP) - the core function of our model.
    """
    _load_models_and_limits()
    asset_type = str(asset_type)
    if asset_type not in _all_models:
        raise ValueError(f"Asset type {asset_type} not found in models file.")
    model = _all_models[asset_type]
    iso = model['iso']
    max_fitted_hp = model['max_hp_fitted']
    slope = model['extrap_slope']
    intercept = model['extrap_intercept']
    max_restriction = model['max_restriction_asset']

    if hp <= max_fitted_hp:
        # Within fitted range - use isotonic model
        restriction = iso.predict([[hp]])[0]
    else:
        # Beyond fitted range - use linear extrapolation
        restriction = slope * hp + intercept

    # Cap at physical limit
    return min(restriction, max_restriction)


# ============================================================================
# INVERSE FUNCTION: HP = R_clean^(-1)(Restriction)
# ============================================================================
# PRESENTATION KEY POINT:
# This inverse function is critical for the clogging calculation algorithm.
# It answers: "What HP would produce this restriction level on a clean filter?"
#
# BENEFITS OF OUR APPROACH:
# - Numerically stable binary search for isotonic regime
# - Analytical solution for linear extrapolation regime
# - Handles edge cases (target beyond model range)
#
# TRADE-OFF:
# Binary search adds computational cost (~100 iterations max) but ensures
# accuracy within 0.01 restriction units

def _get_hp_from_restriction(asset_type, target_restriction):
    """
    Given an asset_type and target restriction, find the HP value on the clean baseline.
    This is the inverse function of R_clean(HP).
    """
    _load_models_and_limits()
    asset_type = str(asset_type)
    if asset_type not in _all_models:
        raise ValueError(f"Asset type {asset_type} not found in models file.")
    model = _all_models[asset_type]
    iso = model['iso']
    max_fitted_hp = model['max_hp_fitted']
    min_fitted_hp = model['min_hp_fitted']
    slope = model['extrap_slope']
    intercept = model['extrap_intercept']
    max_hp = model['max_hp_asset']
    max_restriction = model['max_restriction_asset']

    # Cap target restriction at physical limit
    target_restriction = min(target_restriction, max_restriction)

    # Get max restriction in fitted range
    iso_restriction_max = iso.predict([[max_fitted_hp]])[0]

    if target_restriction <= iso_restriction_max:

        # PRESENTATION NOTE: Binary search to invert isotonic function
        # Isotonic functions are not analytically invertible, so we use
        # numerical root-finding. This is computationally efficient (O(log n))
        # Target restriction is within isotonic fitted range - use binary search
        # The isotonic model maps HP → Restriction, we need to invert it
        hp_min = min_fitted_hp
        hp_max = max_fitted_hp

        # Binary search for the HP that gives target_restriction
        tolerance = 0.01  # Restriction tolerance
        max_iterations = 100

        for iteration in range(max_iterations):
            hp_mid = (hp_min + hp_max) / 2
            restriction_mid = iso.predict([[hp_mid]])[0]

            if abs(restriction_mid - target_restriction) < tolerance:
                return hp_mid

            if restriction_mid < target_restriction:
                hp_min = hp_mid
            else:
                hp_max = hp_mid

        # If we didn't converge, return the midpoint
        return hp_mid
    else:
        # PRESENTATION NOTE: Linear extrapolation allows analytical inversion
        # Target restriction is above fitted range - use inverse of linear extrapolation
        if slope > 0:
            hp_extrap = (target_restriction - intercept) / slope
            return min(hp_extrap, max_hp)  # Cap at max HP
        else:
            # Flat extrapolation - can't reach higher restriction
            return max_hp


# ============================================================================
# MAIN CLOGGING CALCULATION ALGORITHM
# ============================================================================
# PRESENTATION CENTERPIECE:
# This function implements our novel three-step clogging assessment algorithm:
#
# ALGORITHM OVERVIEW:
# Step 1: Calculate delta (excess restriction beyond clean baseline)
#         delta = R_actual - R_clean(HP_current)
#         
# Step 2: Find HP_max_current (maximum HP achievable with current clogging)
#         Solve: R_clean(HP_max_current) + delta = R_max
#         
# Step 3: Convert to percent_clogged
#         percent_clogged = 100 × (1 - HP_max_current / HP_max_asset)
#
# KEY BENEFITS OF THIS APPROACH:
# 1. Physics-grounded: Based on fundamental principle that clogging adds 
#    constant resistance across all HP levels (delta)
# 2. Actionable metric: percent_clogged directly tells operators impact on
#    equipment capability
# 3. Early warning: Detects degradation before reaching critical thresholds
# 4. Asset-specific: Accounts for different baseline curves per equipment type
#
# TRADE-OFFS vs ALTERNATIVE APPROACHES:
# vs Simple Threshold (e.g., "restriction > 25 = bad"):
#   + Better: Context-aware (same restriction means different things at different HP)
#   + Better: Quantifies impact on capability, not just absolute value
#   - Worse: More complex to explain to non-technical users
#
# vs Time-based maintenance (e.g., "change every 6 months"):
#   + Better: Condition-based (change when actually needed)
#   + Better: Prevents both premature and delayed replacements
#   - Worse: Requires sensor infrastructure and monitoring
#
# vs Delta-only approach (just showing restriction increase):
#   + Better: Translates to operational impact (% capability loss)
#   + Better: Normalizes across different asset types
#   - Worse: Requires more computation (inverse function solving)

def calculate_clogging(asset_type, hp, restriction):
    """
    Calculate delta, HP_max_current, and percent_clogged for a new reading.

    Algorithm:
    1. Compute delta = restriction - R_clean(HP) (clip at ≥ 0)
    2. Solve for HP_max_current: the HP where R_clean(HP_max_current) + delta = max_restriction(asset_type)
    3. Compute: percent_clogged = 100 * (1 - HP_max_current / max_horsepower(asset_type))

    Args:
        asset_type (str or int): Asset type (e.g., '2285')
        hp (float): Current hydraulic horsepower reading
        restriction (float): Current air filter restriction reading

    Returns:
        dict: {
            "delta": float,
            "HP_max_current": float,
            "percent_clogged": float
        }
    """
    _load_models_and_limits()
    asset_type = str(asset_type)
    if asset_type not in _limits.index:
        raise ValueError(f"Asset type {asset_type} not found in limits file.")
    if asset_type not in _all_models:
        raise ValueError(f"Asset type {asset_type} not found in models file.")

    max_restriction = float(_limits.loc[asset_type, "Max_AirFilterRestriction"])
    max_horsepower = float(_limits.loc[asset_type, "Max_Horsepower"])
    model = _all_models[asset_type]


    # ========================================================================
    # STEP 1: Calculate delta (clogging signature)
    # ========================================================================
    # PRESENTATION NOTE: Delta is the fundamental measure of filter degradation
    # - If delta = 0: Filter is performing like new (on clean baseline)
    # - If delta > 0: Filter is clogged (adding excess resistance)
    # - Clipped at 0 to handle measurement noise (restriction can't be below clean)

    # Step 1: Calculate delta
    # R_clean(HP) tells us the minimum restriction needed for this HP when clean
    r_clean_hp = _predict_restriction_from_hp(asset_type, hp)
    delta = max(0.0, restriction - r_clean_hp)

    print(f"[DEBUG] Current HP: {hp}, Current Restriction: {restriction}")
    print(f"[DEBUG] R_clean({hp}) = {r_clean_hp:.2f}")
    print(f"[DEBUG] delta = {delta:.3f} (excess restriction beyond clean baseline)")


    # ========================================================================
    # STEP 2: Solve for HP_max_current
    # ========================================================================
    # PRESENTATION KEY POINT: This is where we translate clogging into capability loss
    # We ask: "With this level of clogging (delta), what's the maximum HP we can reach
    # before hitting the equipment's restriction limit?"
    #
    # The equation: R_clean(HP_max_current) + delta = max_restriction
    # Rearranged: R_clean(HP_max_current) = max_restriction - delta
    # 
    # BENEFIT: This directly connects filter condition to operational capacity

    # Step 2: Solve for HP_max_current
    # We need to find HP such that: R_clean(HP) + delta = max_restriction
    # Or equivalently: R_clean(HP) = max_restriction - delta
    target_restriction = max_restriction - delta

    print(f"[DEBUG] max_restriction = {max_restriction}")
    print(f"[DEBUG] target_restriction = max_restriction - delta = {target_restriction:.2f}")

    # Check if target_restriction is achievable
    min_r_possible = model['min_restriction_fitted']
    max_r_possible = _predict_restriction_from_hp(asset_type, max_horsepower)

    print(f"[DEBUG] Model can predict restriction range: {min_r_possible:.2f} to {max_r_possible:.2f}")

    # PRESENTATION NOTE: Edge case handling ensures robustness

    if target_restriction <= min_r_possible:
        # Target restriction is below model's minimum - system can't reach that low even at min HP
        print("[DEBUG] target_restriction below model minimum - using min HP")
        hp_max_current = model['min_hp_fitted']
    elif target_restriction >= max_r_possible:
        # Target restriction is at or above where we'd reach max HP
        print("[DEBUG] target_restriction at or above max HP achievable restriction - using max HP")
        hp_max_current = max_horsepower
    else:
        # Need to solve: find HP such that R_clean(HP) = target_restriction
        # Use the inverse function
        print("[DEBUG] Solving for HP_max_current using inverse function...")
        hp_max_current = _get_hp_from_restriction(asset_type, target_restriction)
        print(f"[DEBUG] HP_max_current = {hp_max_current:.2f}")


    # ========================================================================
    # STEP 3: Calculate percent_clogged
    # ========================================================================
    # PRESENTATION KEY POINT: Converting to percentage makes it actionable
    # - 0% = Filter like new, full capability available
    # - 50% = Half of maximum HP capacity lost to clogging
    # - 100% = Filter completely restricts operation
    #
    # BENEFIT: Universal metric that works across all asset types
    # TRADE-OFF: Linearizes what may be non-linear degradation patterns

    # Step 3: Calculate percent_clogged
    percent_clogged = 100 * (1 - hp_max_current / max_horsepower)
    percent_clogged = np.clip(percent_clogged, 0, 100)

    print(f"[DEBUG] Final: HP_max_current = {hp_max_current:.2f}")
    print(f"[DEBUG] Final: max_horsepower = {max_horsepower}")
    print(f"[DEBUG] Final: percent_clogged = {percent_clogged:.2f}%")
    print()

    return {
        "delta": delta,
        "HP_max_current": hp_max_current,
        "percent_clogged": percent_clogged
    }



# ============================================================================
# SUMMARY FOR PRESENTATION
# ============================================================================
# OVERALL APPROACH BENEFITS:
# 1. Combines strengths of data-driven ML and physics-based modeling
# 2. Provides interpretable, actionable metrics for operators
# 3. Asset-specific calibration ensures accuracy across equipment types
# 4. Robust to edge cases and measurement noise
#
# KEY TRADE-OFFS:
# - Complexity: More sophisticated than simple threshold rules
# - Data dependency: Requires historical clean filter data for training
# - Computational cost: Binary search and model prediction vs simple lookup
# - Assumption: Delta (clogging effect) is constant across HP range
#
# IMPACT:
# - Enables predictive maintenance (replace before failure)
# - Optimizes filter replacement timing (not too early, not too late)
# - Quantifies operational impact (% capability loss)
# - Scales across diverse asset fleet with asset-specific models