/**
 * CardioRisk AI - Frontend Controller
 * Interacts with Flask API for Decision Tree Regression Inference
 */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("riskAssessmentForm");
  const calculateBtn = document.getElementById("calculateBtn");
  
  // Sliders and Indicators
  const bmiSlider = document.getElementById("BMI");
  const bmiDisplay = document.getElementById("bmiDisplay");
  const bmiStatusBadge = document.getElementById("bmiStatusBadge");
  
  const sleepSlider = document.getElementById("SleepHours");
  const sleepDisplay = document.getElementById("sleepDisplay");

  const physSlider = document.getElementById("PhysicalHealthDays");
  const physDisplay = document.getElementById("physDisplay");

  const mentSlider = document.getElementById("MentalHealthDays");
  const mentDisplay = document.getElementById("mentDisplay");

  // BMI Calculator
  const calcHeight = document.getElementById("calcHeight");
  const calcWeight = document.getElementById("calcWeight");
  const applyBmiBtn = document.getElementById("applyBmiBtn");

  // Gauge & Readouts
  const gaugeFillArc = document.getElementById("gaugeProgressArc");
  const gaugeNeedle = document.getElementById("gaugeNeedle");
  const riskPercentDisplay = document.getElementById("riskPercentDisplay");
  const riskBadge = document.getElementById("riskBadge");
  const riskSummaryBox = document.getElementById("riskSummaryBox");
  const riskSummaryText = document.getElementById("riskSummaryText");
  const relativeRiskDisplay = document.getElementById("relativeRiskDisplay");
  const compFillBar = document.getElementById("compFillBar");
  const factorsList = document.getElementById("factorsList");
  const adviceList = document.getElementById("adviceList");

  // Modal
  const openModelMetaBtn = document.getElementById("openModelMetaBtn");
  const closeModalBtn = document.getElementById("closeModalBtn");
  const metaModal = document.getElementById("metaModal");

  // Presets
  const presetButtons = document.querySelectorAll(".preset-btn");

  let currentRiskVal = 0.73;

  /* ==========================================================================
     Slider Listeners & UI Helpers
     ========================================================================== */
  function updateBmiStatus(val) {
    bmiDisplay.textContent = `${parseFloat(val).toFixed(1)} kg/m²`;
    const num = parseFloat(val);
    if (num < 18.5) {
      bmiStatusBadge.textContent = `Underweight (${num.toFixed(1)})`;
      bmiStatusBadge.style.color = "#0284c7";
      bmiStatusBadge.style.borderColor = "#bae6fd";
    } else if (num < 25.0) {
      bmiStatusBadge.textContent = `Normal (${num.toFixed(1)})`;
      bmiStatusBadge.style.color = "#15803d";
      bmiStatusBadge.style.borderColor = "#bbf7d0";
    } else if (num < 30.0) {
      bmiStatusBadge.textContent = `Overweight (${num.toFixed(1)})`;
      bmiStatusBadge.style.color = "#b45309";
      bmiStatusBadge.style.borderColor = "#fde68a";
    } else {
      bmiStatusBadge.textContent = `Obese (${num.toFixed(1)})`;
      bmiStatusBadge.style.color = "#b91c1c";
      bmiStatusBadge.style.borderColor = "#fca5a5";
    }
  }

  bmiSlider.addEventListener("input", (e) => updateBmiStatus(e.target.value));
  sleepSlider.addEventListener("input", (e) => {
    sleepDisplay.textContent = `${e.target.value} Hours`;
  });
  physSlider.addEventListener("input", (e) => {
    physDisplay.textContent = `${e.target.value} Days`;
  });
  mentSlider.addEventListener("input", (e) => {
    mentDisplay.textContent = `${e.target.value} Days`;
  });

  // Calculate BMI Helper
  applyBmiBtn.addEventListener("click", () => {
    const h = parseFloat(calcHeight.value) / 100; // to meters
    const w = parseFloat(calcWeight.value);
    if (h > 0 && w > 0) {
      const calculatedBmi = Math.min(55, Math.max(15, (w / (h * h)))).toFixed(1);
      bmiSlider.value = calculatedBmi;
      updateBmiStatus(calculatedBmi);
      // Auto trigger evaluation
      triggerPrediction();
    }
  });

  /* ==========================================================================
     Gauge Animation
     ========================================================================== */
  function setGaugeValue(percent, color) {
    const totalLength = 283; // SVG Semicircle circumference for r=90
    const clamped = Math.max(0, Math.min(100, percent));
    
    // Calculate offset
    const offset = totalLength - (clamped / 100) * totalLength;
    gaugeFillArc.style.strokeDashoffset = offset;
    gaugeFillArc.style.stroke = color || "#e11d48";

    // Needle angle: -90deg is 0%, +90deg is 100%
    const angle = -90 + (clamped / 100) * 180;
    gaugeNeedle.style.transform = `rotate(${angle}deg)`;

    // Comparison bar fill
    compFillBar.style.width = `${Math.min(100, Math.max(2, clamped))}%`;
    compFillBar.style.background = color || "#e11d48";
  }

  function animateRiskNumber(targetVal) {
    const startVal = currentRiskVal;
    const duration = 600;
    const startTime = performance.now();

    function updateCounter(now) {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      // Ease-out cubic
      const ease = 1 - Math.pow(1 - progress, 3);
      const current = startVal + (targetVal - startVal) * ease;
      riskPercentDisplay.innerHTML = `${current.toFixed(2)}<span class="percent-sign">%</span>`;

      if (progress < 1) {
        requestAnimationFrame(updateCounter);
      } else {
        currentRiskVal = targetVal;
      }
    }
    requestAnimationFrame(updateCounter);
  }

  /* ==========================================================================
     Collect Data & Make API Request
     ========================================================================== */
  function collectFormData() {
    const formData = new FormData(form);
    const data = {};

    // Selects and text/numbers
    for (const [key, value] of formData.entries()) {
      data[key] = value;
    }

    // Explicitly handle checkboxes (Yes / No)
    const binaryCheckboxes = [
      "HadAngina", "HadStroke", "HadCOPD", "HadKidneyDisease", 
      "ChestScan", "DifficultyWalking", "HadArthritis", 
      "PhysicalActivities", "AlcoholDrinkers", "HadAsthma"
    ];

    binaryCheckboxes.forEach((id) => {
      const el = document.getElementById(id);
      if (el) {
        data[id] = el.checked ? "Yes" : "No";
      }
    });

    // Handle Radios (Sex)
    const sexRadios = document.querySelectorAll('input[name="Sex"]');
    for (const r of sexRadios) {
      if (r.checked) {
        data["Sex"] = r.value;
        break;
      }
    }

    // Explicit numeric conversions
    data["BMI"] = parseFloat(bmiSlider.value);
    data["PhysicalHealthDays"] = parseFloat(physSlider.value);
    data["MentalHealthDays"] = parseFloat(mentSlider.value);
    data["SleepHours"] = parseFloat(sleepSlider.value);

    return data;
  }

  async function triggerPrediction() {
    const payload = collectFormData();
    calculateBtn.disabled = true;
    calculateBtn.style.opacity = "0.75";
    calculateBtn.querySelector(".btn-text").textContent = "Evaluating Cardiac Model...";

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const result = await res.json();

      // Animate Gauge & Readout
      setGaugeValue(result.risk_percent, result.color);
      animateRiskNumber(result.risk_percent);

      // Badge
      riskBadge.className = `risk-badge ${result.badge_class}`;
      riskBadge.textContent = result.tier.toUpperCase();

      // Comparison text
      relativeRiskDisplay.innerHTML = `<strong>${result.relative_to_baseline}×</strong> average baseline`;

      // Summary text
      riskSummaryText.textContent = result.summary;
      riskSummaryBox.style.borderLeftColor = result.color;

      // Identified Factors
      if (result.factors && result.factors.length > 0) {
        factorsList.innerHTML = result.factors.map(f => `
          <div class="factor-item">
            <div>
              <div class="name">${f.name}</div>
              <div class="desc">${f.desc}</div>
            </div>
            <span class="factor-badge">${f.impact}</span>
          </div>
        `).join("");
      } else {
        factorsList.innerHTML = `<div class="empty-factors">✓ No elevated cardiovascular risk drivers detected in this profile.</div>`;
      }

      // Guidance / Advice
      if (result.advice && result.advice.length > 0) {
        adviceList.innerHTML = result.advice.map(item => `<li>${item}</li>`).join("");
      }

    } catch (err) {
      console.error("Prediction error:", err);
      riskSummaryText.textContent = "Error communicating with the prediction engine: " + err.message;
    } finally {
      calculateBtn.disabled = false;
      calculateBtn.style.opacity = "1";
      calculateBtn.querySelector(".btn-text").textContent = "Predict Heart Attack Risk";
    }
  }

  // Handle Form Submit
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    triggerPrediction();
  });

  /* ==========================================================================
     Preset Profiles
     ========================================================================== */
  const presets = {
    healthy: {
      AgeCategory: "Age 25 to 29",
      Sex: "Female",
      GeneralHealth: "Excellent",
      BMI: 21.5,
      PhysicalActivities: true,
      SmokerStatus: "Never smoked",
      LastCheckupTime: "Within past year (anytime less than 12 months ago)",
      HadAngina: false,
      HadStroke: false,
      HadCOPD: false,
      HadKidneyDisease: false,
      ChestScan: false,
      DifficultyWalking: false,
      HadArthritis: false,
      HadDiabetes: "No",
      HadDepressiveDisorder: "No",
      AlcoholDrinkers: false,
      HadAsthma: false,
      SleepHours: 8,
      PhysicalHealthDays: 0,
      MentalHealthDays: 0,
    },
    moderate: {
      AgeCategory: "Age 65 to 69",
      Sex: "Male",
      GeneralHealth: "Fair",
      BMI: 29.5,
      PhysicalActivities: false,
      SmokerStatus: "Current smoker - now smokes every day",
      LastCheckupTime: "Within past year (anytime less than 12 months ago)",
      HadAngina: false,
      HadStroke: false,
      HadCOPD: false,
      HadKidneyDisease: false,
      ChestScan: true,
      DifficultyWalking: true,
      HadArthritis: true,
      HadDiabetes: "Yes",
      HadDepressiveDisorder: false,
      AlcoholDrinkers: false,
      HadAsthma: false,
      SleepHours: 6.0,
      PhysicalHealthDays: 10,
      MentalHealthDays: 5,
    },
    high: {
      AgeCategory: "Age 70 to 74",
      Sex: "Male",
      GeneralHealth: "Poor",
      BMI: 33.2,
      PhysicalActivities: false,
      SmokerStatus: "Current smoker - now smokes every day",
      LastCheckupTime: "Within past year (anytime less than 12 months ago)",
      HadAngina: true,
      HadStroke: true,
      HadCOPD: true,
      HadKidneyDisease: true,
      ChestScan: true,
      DifficultyWalking: true,
      HadArthritis: true,
      HadDiabetes: "Yes",
      HadDepressiveDisorder: "Yes",
      AlcoholDrinkers: false,
      HadAsthma: true,
      SleepHours: 5,
      PhysicalHealthDays: 15,
      MentalHealthDays: 10,
    },
    default: {
      AgeCategory: "Age 50 to 54",
      Sex: "Male",
      GeneralHealth: "Very good",
      BMI: 24.2,
      PhysicalActivities: true,
      SmokerStatus: "Never smoked",
      LastCheckupTime: "Within past year (anytime less than 12 months ago)",
      HadAngina: false,
      HadStroke: false,
      HadCOPD: false,
      HadKidneyDisease: false,
      ChestScan: false,
      DifficultyWalking: false,
      HadArthritis: false,
      HadDiabetes: "No",
      HadDepressiveDisorder: "No",
      AlcoholDrinkers: false,
      HadAsthma: false,
      SleepHours: 7,
      PhysicalHealthDays: 0,
      MentalHealthDays: 0,
    }
  };

  function applyPreset(presetKey) {
    const config = presets[presetKey];
    if (!config) return;

    // Apply values to inputs
    for (const [key, val] of Object.entries(config)) {
      if (typeof val === "boolean") {
        const el = document.getElementById(key);
        if (el) el.checked = val;
      } else if (key === "Sex") {
        const radio = document.querySelector(`input[name="Sex"][value="${val}"]`);
        if (radio) radio.checked = true;
      } else {
        const el = document.getElementById(key);
        if (el) el.value = val;
      }
    }

    // Refresh display values
    updateBmiStatus(config.BMI);
    sleepDisplay.textContent = `${config.SleepHours} Hours`;
    physDisplay.textContent = `${config.PhysicalHealthDays} Days`;
    mentDisplay.textContent = `${config.MentalHealthDays} Days`;

    // Immediately trigger prediction
    triggerPrediction();
  }

  presetButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const presetKey = btn.getAttribute("data-preset");
      applyPreset(presetKey);
    });
  });

  /* ==========================================================================
     Modal Events
     ========================================================================== */
  openModelMetaBtn.addEventListener("click", () => {
    metaModal.style.display = "flex";
  });

  closeModalBtn.addEventListener("click", () => {
    metaModal.style.display = "none";
  });

  metaModal.addEventListener("click", (e) => {
    if (e.target === metaModal) {
      metaModal.style.display = "none";
    }
  });

  // Initial Calculation on Page Load
  updateBmiStatus(bmiSlider.value);
  triggerPrediction();
});
