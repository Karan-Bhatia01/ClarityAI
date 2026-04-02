/* ────────────────────────────────────────────────────────────────
   MODERN UI ENHANCEMENTS — Animations, interactions, smooth flows
   ──────────────────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  // Smooth transitions on page load
  document.body.style.opacity = "1";
  animateElements();
  setupFormInteractions();
  setupScrollEffects();
  setupPageTransitions();
});

/* ────────────── Animate Elements on Load ────────────── */

function animateElements() {
  const elements = document.querySelectorAll(".section, .card, .chart-card");
  elements.forEach((el, index) => {
    el.style.opacity = "0";
    el.style.transform = "translateY(20px)";
    setTimeout(() => {
      el.style.transition = "all 0.6s cubic-bezier(0.4, 0, 0.2, 1)";
      el.style.opacity = "1";
      el.style.transform = "translateY(0)";
    }, index * 50);
  });
}

/* ────────────── Form Interactions ────────────── */

function setupFormInteractions() {
  // Form input focus effects
  const inputs = document.querySelectorAll(
    'input[type="text"], input[type="email"], input[type="password"], input[type="number"], textarea, select'
  );

  inputs.forEach((input) => {
    input.addEventListener("focus", () => {
      input.parentElement.style.transition = "all 0.3s ease";
      input.parentElement.style.transform = "translateY(-1px)";
    });

    input.addEventListener("blur", () => {
      input.parentElement.style.transform = "translateY(0)";
    });

    // Live validation feedback
    input.addEventListener("input", () => {
      if (input.value) {
        input.style.borderColor = "rgba(79, 70, 229, 0.5)";
      } else {
        input.style.borderColor = "";
      }
    });
  });

  // Checkbox animations
  const checkboxes = document.querySelectorAll('input[type="checkbox"]');
  checkboxes.forEach((checkbox) => {
    const label = checkbox.closest(".checkbox-item");
    checkbox.addEventListener("change", () => {
      label.style.transform = checkbox.checked ? "scale(1.02)" : "scale(1)";
      label.style.transition = "transform 0.2s ease";
    });
  });

  // Form submission animation
  const forms = document.querySelectorAll("form");
  forms.forEach((form) => {
    form.addEventListener("submit", (e) => {
      const button = form.querySelector('button[type="submit"]');
      if (button) {
        button.innerHTML = '<span class="loading"></span>';
        button.disabled = true;
      }
    });
  });
}

/* ────────────── Scroll Effects ────────────── */

function setupScrollEffects() {
  // Sticky header animation
  const header = document.querySelector(".page-header");
  let lastScrollY = window.scrollY;

  window.addEventListener("scroll", () => {
    if (header) {
      if (window.scrollY > 50) {
        header.style.boxShadow = "0 4px 12px rgba(0, 0, 0, 0.4)";
      } else {
        header.style.boxShadow = "none";
      }
    }

    // Trigger animations for elements entering viewport
    const elements = document.querySelectorAll(".card, .chart-card, .insights-panel");
    elements.forEach((el) => {
      const rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight * 0.75 && !el.classList.contains("animated")) {
        el.classList.add("animate-in");
        el.classList.add("animated");
      }
    });

    lastScrollY = window.scrollY;
  });
}

/* ────────────── Smooth Page Transitions ────────────── */

function setupPageTransitions() {
  // Add fade-out effect when navigating
  const links = document.querySelectorAll('a[href*="/"]');
  links.forEach((link) => {
    if (!link.href.includes("#")) {
      link.addEventListener("click", (e) => {
        if (!link.target.includes("_blank")) {
          e.preventDefault();
          fadeOutPage(() => {
            window.location.href = link.href;
          });
        }
      });
    }
  });
}

function fadeOutPage(callback) {
  const body = document.body;
  body.style.transition = "opacity 0.3s ease";
  body.style.opacity = "0";
  setTimeout(callback, 300);
}

/* ────────────── Button Hover Effects ────────────── */

const style = document.createElement("style");
style.innerHTML = `
  .btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: rgba(255, 255, 255, 0.1);
    transition: left 0.5s ease;
    z-index: 1;
    pointer-events: none;
  }

  .btn:hover::before {
    left: 100%;
  }

  .btn-primary::before {
    background: rgba(255, 255, 255, 0.2);
  }

  .btn,
  .btn-ai-insights {
    position: relative;
  }
`;
document.head.appendChild(style);

/* ────────────── Loading State ────────────── */

function showLoader() {
  const loader = document.createElement("div");
  loader.className = "loading-overlay";
  loader.innerHTML = '<div class="loading"></div>';
  document.body.appendChild(loader);
  return loader;
}

function hideLoader(loader) {
  if (loader) {
    loader.style.opacity = "0";
    loader.style.transition = "opacity 0.3s ease";
    setTimeout(() => loader.remove(), 300);
  }
}

/* ────────────── Toast Notifications ────────────── */

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.style.cssText = `
    position: fixed;
    bottom: 2rem;
    right: 2rem;
    padding: 1rem 1.5rem;
    background-color: ${
      type === "success"
        ? "rgb(16, 185, 129)"
        : type === "error"
          ? "rgb(239, 68, 68)"
          : "rgb(79, 70, 229)"
    };
    color: white;
    border-radius: 0.5rem;
    font-size: 0.875rem;
    font-weight: 500;
    z-index: 2000;
    animation: slideUp 0.3s ease-out;
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
  `;
  toast.textContent = message;
  document.body.appendChild(toast);

  setTimeout(() => {
    toast.style.animation = "fadeOut 0.3s ease-out";
    setTimeout(() => toast.remove(), 300);
  }, 3000);
}

/* ────────────── Modal Dialogs ────────────── */

function createModal(title, content, actions = []) {
  const modal = document.createElement("div");
  modal.style.cssText = `
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: rgba(0, 0, 0, 0.5);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 1000;
    backdrop-filter: blur(4px);
    animation: fadeIn 0.3s ease-out;
  `;

  const dialog = document.createElement("div");
  dialog.style.cssText = `
    background-color: var(--bg-secondary);
    border: 1px solid var(--border-color);
    border-radius: var(--radius-lg);
    padding: 2rem;
    max-width: 400px;
    width: 90%;
    animation: slideUp 0.3s ease-out;
  `;

  let html = `<h2 style="margin-top: 0; color: var(--fg-primary);">${title}</h2>`;
  html += `<p style="color: var(--fg-secondary);">${content}</p>`;
  html += '<div style="display: flex; gap: 1rem; margin-top: 2rem;">';

  actions.forEach((action) => {
    html += `<button class="btn btn-${action.type || 'secondary'}" style="flex: 1;">${action.label}</button>`;
  });

  html += "</div>";
  dialog.innerHTML = html;
  modal.appendChild(dialog);

  const buttons = dialog.querySelectorAll("button");
  buttons.forEach((btn, index) => {
    btn.addEventListener("click", () => {
      if (actions[index] && actions[index].callback) {
        actions[index].callback();
      }
      closeModal(modal);
    });
  });

  document.body.appendChild(modal);
  return modal;
}

function closeModal(modal) {
  modal.style.animation = "fadeOut 0.3s ease-out";
  setTimeout(() => modal.remove(), 300);
}

/* ────────────── Keyboard Shortcuts ────────────── */

document.addEventListener("keydown", (e) => {
  // Escape to close modals
  if (e.key === "Escape") {
    const modals = document.querySelectorAll('[role="dialog"]');
    modals.forEach((modal) => modal.remove());
  }

  // Ctrl/Cmd + Enter to submit forms
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    const form = document.querySelector("form");
    if (form) {
      form.dispatchEvent(new Event("submit", { cancelable: true }));
      form.submit();
    }
  }
});

/* ────────────── Accessibility Enhancements ────────────── */

// High contrast mode detection
const darkModeQuery = window.matchMedia("(prefers-color-scheme: dark)");
darkModeQuery.addListener((e) => {
  document.documentElement.style.colorScheme = e.matches ? "dark" : "light";
});

// Reduced motion
const prefersReducedMotion = window.matchMedia(
  "(prefers-reduced-motion: reduce)"
);
if (prefersReducedMotion.matches) {
  document.documentElement.style.scrollBehavior = "auto";
  document.documentElement.style.setProperty("--transition-fast", "0ms");
  document.documentElement.style.setProperty("--transition-base", "0ms");
  document.documentElement.style.setProperty("--transition-slow", "0ms");
}

/* ────────────── Export Functions for Global Use ────────────── */

window.showToast = showToast;
window.showLoader = showLoader;
window.hideLoader = hideLoader;
window.createModal = createModal;
window.closeModal = closeModal;
