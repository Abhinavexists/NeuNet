// ========================================
// UNIFIED NEUNET WEBSITE FUNCTIONALITY
// ========================================

class NeuNetWebsite {
  constructor() {
    // Ensure DOM is ready before initializing
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', () => this.init());
    } else {
      this.init();
    }
  }

  init() {
    try {
      this.setupMobileMenu();
      this.setupSmoothScroll();
      this.setupScrollEffects();
      this.setupAnimations();
      this.setupActiveNavigation();
      this.setupCodeCopy();
      this.setupThemeToggle();
    } catch (error) {
      console.warn('NeuNet Website initialization error:', error);
      // Ensure content is visible even if scripts fail
      this.ensureContentVisibility();
    }
  }

  // Fallback to ensure content is always visible
  ensureContentVisibility() {
    try {
      document.querySelectorAll('.fade-in').forEach(el => {
        el.style.opacity = '1';
        el.style.transform = 'translateY(0)';
      });
      document.querySelectorAll('.card').forEach(card => {
        card.style.opacity = '1';
        card.style.transform = 'translateY(0)';
      });
    } catch (error) {
      console.warn('Error ensuring content visibility:', error);
    }
  }

  // Mobile Menu Functionality
  setupMobileMenu() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (!hamburger || !navMenu) return;

    hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
      
      // Update aria-expanded for accessibility
      const isExpanded = navMenu.classList.contains('active');
      hamburger.setAttribute('aria-expanded', isExpanded);
        });
        
        // Close menu when clicking on a link
    navMenu.querySelectorAll('a').forEach(link => {
            link.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
        hamburger.setAttribute('aria-expanded', 'false');
            });
        });

    // Close menu when clicking outside
    document.addEventListener('click', (e) => {
      if (!hamburger.contains(e.target) && !navMenu.contains(e.target)) {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
        hamburger.setAttribute('aria-expanded', 'false');
      }
    });
  }

  // Smooth Scrolling for Anchor Links
  setupSmoothScroll() {
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
      anchor.addEventListener('click', (e) => {
        e.preventDefault();
        const target = document.querySelector(anchor.getAttribute('href'));
        
        if (target) {
          const headerOffset = 100;
          const elementPosition = target.getBoundingClientRect().top;
          const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

          window.scrollTo({
            top: offsetPosition,
            behavior: 'smooth'
            });
        }
    });
});
  }

// Navbar Background on Scroll
  setupScrollEffects() {
    const navbar = document.querySelector('.navbar');
    if (!navbar) return;

    const updateNavbar = () => {
      if (window.scrollY > 50) {
        navbar.style.background = 'rgba(15, 23, 42, 0.98)';
        navbar.style.boxShadow = 'var(--shadow-lg)';
    } else {
        navbar.style.background = 'rgba(15, 23, 42, 0.95)';
        navbar.style.boxShadow = 'none';
      }
    };

    window.addEventListener('scroll', updateNavbar);
    updateNavbar(); // Initial call
  }

  // Scroll-triggered Animations
  setupAnimations() {
    // Check if IntersectionObserver is supported
    if (!window.IntersectionObserver) {
      // Fallback: just show all elements immediately
      document.querySelectorAll('.fade-in').forEach(el => {
        el.classList.add('visible');
      });
      return;
    }

const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

    const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('visible');
          // For cards with custom animation, ensure they're visible
          if (entry.target.classList.contains('card') && entry.target.style.opacity === '0') {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
          }
        }
    });
}, observerOptions);

    // Handle fade-in elements
    document.querySelectorAll('.fade-in').forEach(el => {
      // Only apply fade-in animation to elements that are initially off-screen
      const rect = el.getBoundingClientRect();
      const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
      
      if (!isVisible) {
        // Remove the visible class so the CSS animation can work
        el.classList.remove('visible');
        observer.observe(el);
      } else {
        // Elements already visible should stay visible
        el.classList.add('visible');
      }
    });

    // Setup cards for staggered animation - but make them visible by default
    document.querySelectorAll('.card').forEach((card, index) => {
      // Only apply animation if the card is not currently visible on screen
      const rect = card.getBoundingClientRect();
      const isVisible = rect.top < window.innerHeight && rect.bottom > 0;
      
      if (!isVisible) {
        // Only hide and animate cards that are off-screen
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = `opacity 0.6s ease ${index * 0.1}s, transform 0.6s ease ${index * 0.1}s`;
        observer.observe(card);
      }
      // Note: Cards already visible will stay visible due to CSS defaults
    });

    // Reduced fallback time since elements are visible by default
    setTimeout(() => {
      document.querySelectorAll('.fade-in:not(.visible)').forEach(el => {
        el.classList.add('visible');
      });
      document.querySelectorAll('.card').forEach(card => {
        if (card.style.opacity === '0') {
          card.style.opacity = '1';
          card.style.transform = 'translateY(0)';
        }
      });
    }, 1000); // Reduced from 2000ms to 1000ms
  }

  // Active Navigation State
  setupActiveNavigation() {
    const currentPage = window.location.pathname.split('/').pop() || 'index.html';
    
    document.querySelectorAll('.nav-menu a, .sidebar-nav a').forEach(link => {
      const href = link.getAttribute('href');
      
      if (href === currentPage || (currentPage === '' && href === 'index.html')) {
        link.classList.add('active');
                } else {
        link.classList.remove('active');
      }
    });

    // Highlight sidebar sections on scroll
    if (document.querySelector('.sidebar-nav')) {
      this.setupSidebarHighlight();
    }
  }

  // Sidebar Section Highlighting
  setupSidebarHighlight() {
    const sections = document.querySelectorAll('main [id]');
    const navLinks = document.querySelectorAll('.sidebar-nav a[href^="#"]');
    
    if (sections.length === 0 || navLinks.length === 0) return;

    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        const id = entry.target.getAttribute('id');
        const navLink = document.querySelector(`.sidebar-nav a[href="#${id}"]`);
        
        if (navLink) {
          if (entry.isIntersecting) {
            navLinks.forEach(link => link.classList.remove('active'));
            navLink.classList.add('active');
          }
        }
      });
    }, {
      rootMargin: '-100px 0px -50% 0px',
      threshold: 0
    });

    sections.forEach(section => observer.observe(section));
  }

  // Code Copy Functionality
  setupCodeCopy() {
    document.querySelectorAll('pre code').forEach(block => {
        const button = document.createElement('button');
        button.className = 'copy-btn';
        button.textContent = 'Copy';
        button.style.cssText = `
            position: absolute;
        top: var(--space-3);
        right: var(--space-3);
        background: var(--color-primary);
        color: var(--color-primary-text);
            border: none;
        padding: var(--space-2) var(--space-3);
        border-radius: var(--radius-md);
        font-size: var(--text-xs);
        font-weight: var(--font-weight-medium);
            cursor: pointer;
            opacity: 0;
        transition: all var(--transition-fast);
        z-index: 10;
        `;
        
        const pre = block.parentElement;
        pre.style.position = 'relative';
        pre.appendChild(button);
        
        // Show/hide copy button on hover
        pre.addEventListener('mouseenter', () => {
            button.style.opacity = '1';
        });
        
        pre.addEventListener('mouseleave', () => {
            button.style.opacity = '0';
        });
        
        // Copy functionality
      button.addEventListener('click', async () => {
        try {
          await navigator.clipboard.writeText(block.textContent);
                button.textContent = 'Copied!';
          button.style.background = 'var(--color-success)';
          
                setTimeout(() => {
                    button.textContent = 'Copy';
            button.style.background = 'var(--color-primary)';
                }, 2000);
        } catch (err) {
          console.error('Failed to copy text: ', err);
        }
        });
    });
  }

  // Theme Toggle (for future light mode support)
  setupThemeToggle() {
    // This can be implemented later for light/dark mode switching
    const themeToggle = document.querySelector('.theme-toggle');
    if (!themeToggle) return;

    themeToggle.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
      
      document.documentElement.setAttribute('data-theme', newTheme);
      localStorage.setItem('theme', newTheme);
    });

    // Load saved theme
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-theme', savedTheme);
  }

  // Counter Animation for Stats
  static animateCounter(element, target, duration = 2000) {
    let startTime = null;
    const startValue = 0;
    
    const updateCounter = (currentTime) => {
      if (!startTime) startTime = currentTime;
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      
      // Easing function
      const easeOutCubic = 1 - Math.pow(1 - progress, 3);
      const currentValue = startValue + (target - startValue) * easeOutCubic;
      
      if (element.textContent.includes('%')) {
        element.textContent = Math.round(currentValue) + '%';
      } else if (element.textContent.includes('+')) {
        element.textContent = Math.round(currentValue) + '+';
      } else {
        element.textContent = Math.round(currentValue);
      }
      
      if (progress < 1) {
        requestAnimationFrame(updateCounter);
      }
    };
    
    requestAnimationFrame(updateCounter);
  }
}

// Initialize when DOM is ready
function initializeWebsite() {
  try {
    // Initialize the main website class
    const website = new NeuNetWebsite();
    
    // Set up counter animations
    const statsObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          try {
            const statNumbers = entry.target.querySelectorAll('[style*="font-size: var(--text-4xl)"]');
            
            statNumbers.forEach(stat => {
              const text = stat.textContent;
              let target;
              
              if (text.includes('%')) {
                target = parseInt(text.replace('%', ''));
              } else if (text.includes('+')) {
                target = parseInt(text.replace('+', ''));
              } else {
                target = parseInt(text);
              }
              
              if (!isNaN(target)) {
                NeuNetWebsite.animateCounter(stat, target);
              }
            });
            
            statsObserver.unobserve(entry.target);
          } catch (error) {
            console.warn('Error in stats animation:', error);
          }
        }
      });
    }, { threshold: 0.5 });

    // Observe stats sections
    try {
      document.querySelectorAll('.grid').forEach(grid => {
        if (grid.querySelector('[style*="font-size: var(--text-4xl)"]')) {
          statsObserver.observe(grid);
        }
      });
    } catch (error) {
      console.warn('Error setting up stats observer:', error);
    }
    
  } catch (error) {
    console.error('Failed to initialize website:', error);
    // Ensure content is visible as fallback
    try {
      const elements = document.querySelectorAll('.fade-in, .card');
      elements.forEach(el => {
        el.style.opacity = '1';
        el.style.transform = 'translateY(0)';
      });
    } catch (fallbackError) {
      console.error('Even fallback failed:', fallbackError);
    }
  }
}

// Multiple initialization strategies
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initializeWebsite);
} else {
  // DOM is already loaded
  initializeWebsite();
}

// Additional fallback - ensure content is visible after a short delay
setTimeout(() => {
  try {
    const hiddenElements = document.querySelectorAll('.fade-in[style*="opacity: 0"], .card[style*="opacity: 0"]');
    if (hiddenElements.length > 0) {
      console.warn('Found hidden elements after timeout, making them visible');
      hiddenElements.forEach(el => {
        el.style.opacity = '1';
        el.style.transform = 'translateY(0)';
      });
    }
  } catch (error) {
    console.error('Error in fallback visibility check:', error);
  }
}, 2000);

// Handle keyboard navigation
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') {
    try {
      // Close mobile menu
      const hamburger = document.querySelector('.hamburger');
      const navMenu = document.querySelector('.nav-menu');
      
      if (hamburger && navMenu && navMenu.classList.contains('active')) {
        hamburger.classList.remove('active');
        navMenu.classList.remove('active');
        hamburger.setAttribute('aria-expanded', 'false');
      }
    } catch (error) {
      console.warn('Error handling Escape key:', error);
    }
  }
}); 