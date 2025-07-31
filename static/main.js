/**
 * JavaScript functionality for annotation interface
 */

// Toggle logic for ChooseOne buttons (radio-style)
function toggleRadio(btn) {
    const target = btn.dataset.target;
    const value = btn.dataset.value;
    
    // Deactivate all sibling buttons
    const siblings = btn.parentElement.querySelectorAll('.choice');
    siblings.forEach(b => {
        b.classList.remove('active');
    });
    
    // Activate clicked button
    btn.classList.add('active');
    
    // Update hidden input with selected value
    const hiddenInput = document.getElementById(`input-${target}`);
    if (hiddenInput) {
        hiddenInput.value = value;
    }
}

// Toggle logic for ChooseMany buttons (checkbox-style)
function toggleCheckbox(btn) {
    const target = btn.dataset.target;
    const value = btn.dataset.value;
    
    // Toggle active state
    btn.classList.toggle('active');
    
    // Collect all active values
    const activeButtons = btn.parentElement.querySelectorAll('.choice.active');
    const values = Array.from(activeButtons).map(b => b.dataset.value);
    
    // Update hidden input with comma-separated values
    const hiddenInput = document.getElementById(`input-${target}`);
    if (hiddenInput) {
        hiddenInput.value = values.join(',');
    }
}

// Flag to track if we're submitting the form
let isSubmitting = false;

// Preserve form values across page loads
document.addEventListener('DOMContentLoaded', function() {
    // Check if annotator name is stored in localStorage
    const storedAnnotator = localStorage.getItem('annotator');
    const annotatorInput = document.querySelector('input[name="annotator"]');
    
    if (storedAnnotator && annotatorInput && !annotatorInput.value) {
        annotatorInput.value = storedAnnotator;
    }
    
    // Save annotator name when it changes
    if (annotatorInput) {
        annotatorInput.addEventListener('change', function() {
            localStorage.setItem('annotator', this.value);
        });
    }
    
    // Restore unfilled task search index
    const storedStartIndex = localStorage.getItem('unfilled-start-index');
    const startIndexInput = document.getElementById('unfilled-start-index');
    
    if (storedStartIndex && startIndexInput) {
        startIndexInput.value = storedStartIndex;
    }
    
    // Save unfilled task search index when it changes
    if (startIndexInput) {
        startIndexInput.addEventListener('change', function() {
            localStorage.setItem('unfilled-start-index', this.value);
        });
        startIndexInput.addEventListener('input', function() {
            localStorage.setItem('unfilled-start-index', this.value);
        });
    }
    
    // Restore unfilled scope radio selection
    const storedScope = localStorage.getItem('unfilled-scope');
    if (storedScope) {
        const scopeRadio = document.querySelector(`input[name="unfilled-scope"][value="${storedScope}"]`);
        if (scopeRadio) {
            scopeRadio.checked = true;
        }
    }
    
    // Save unfilled scope when it changes
    const scopeRadios = document.querySelectorAll('input[name="unfilled-scope"]');
    scopeRadios.forEach(radio => {
        radio.addEventListener('change', function() {
            if (this.checked) {
                localStorage.setItem('unfilled-scope', this.value);
            }
        });
    });
    
    // Add form submit handler to set flag
    const form = document.getElementById('annotation-form');
    if (form) {
        form.addEventListener('submit', function() {
            isSubmitting = true;
        });
    }
    
    // Keyboard shortcuts
    document.addEventListener('keydown', function(e) {
        // Enter key to submit (when not in textarea)
        if (e.key === 'Enter' && document.activeElement.tagName !== 'TEXTAREA') {
            e.preventDefault();
            const submitBtn = document.querySelector('.next-button');
            if (submitBtn) {
                submitBtn.click();
            }
        }
        
        // Number keys for quick selection (if ChooseOne with numeric options)
        if (e.key >= '0' && e.key <= '9') {
            // Only if not typing in a text field
            if (document.activeElement.tagName !== 'INPUT' && 
                document.activeElement.tagName !== 'TEXTAREA') {
                const buttons = document.querySelectorAll('.choice');
                buttons.forEach(btn => {
                    if (btn.textContent.trim() === e.key) {
                        btn.click();
                    }
                });
            }
        }
    });
    
    // Auto-focus on first required empty field
    const requiredInputs = document.querySelectorAll('input[required], .choice-group');
    for (let elem of requiredInputs) {
        if (elem.tagName === 'INPUT' && !elem.value) {
            elem.focus();
            break;
        }
    }
});

// Prevent accidental navigation away (but not when submitting)
window.addEventListener('beforeunload', function(e) {
    // Don't show warning if we're submitting the form
    if (isSubmitting) {
        return;
    }
    
    const form = document.getElementById('annotation-form');
    if (form) {
        // Check if any options have been selected
        const hasSelections = form.querySelector('.choice.active') || 
                            form.querySelector('textarea[name="notes"]').value ||
                            form.querySelector('input[name="unsure"]').checked;
        
        if (hasSelections) {
            e.preventDefault();
            e.returnValue = '';
        }
    }
});