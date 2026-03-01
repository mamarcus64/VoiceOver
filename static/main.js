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
    
    // Check for auto-submit
    checkAutoSubmit();
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
    
    // Check for auto-submit
    checkAutoSubmit();
}

// Flag to track if we're submitting the form
let isSubmitting = false;

// Check if all labels have values and auto-submit if enabled
function checkAutoSubmit() {
    // Check if auto-submit is enabled
    const autoSubmitCheckbox = document.getElementById('auto-submit');
    if (!autoSubmitCheckbox || !autoSubmitCheckbox.checked) {
        return;
    }
    
    // Check if all labels have values
    if (areAllLabelsSet()) {
        // Auto-submit the form
        const submitBtn = document.querySelector('.next-button');
        if (submitBtn && validateForm()) {
            submitBtn.click();
        }
    }
}

// Check if all labels have values
function areAllLabelsSet() {
    const form = document.getElementById('annotation-form');
    if (!form) return false;
    
    // Get all label blocks
    const labelBlocks = form.querySelectorAll('.label-block');
    
    for (let block of labelBlocks) {
        const label = block.querySelector('.option-label');
        if (!label) continue;
        
        // Check ChooseOne and ChooseMany
        const choiceGroup = block.querySelector('.choice-group');
        if (choiceGroup) {
            const hiddenInput = choiceGroup.querySelector('input[type="hidden"]');
            if (hiddenInput && !hiddenInput.value) {
                return false;
            }
        }
        
        // Check FreeText
        const textInput = block.querySelector('input[type="text"]');
        if (textInput && !textInput.value.trim()) {
            return false;
        }
    }
    
    return true;
}

// Preserve form values across page loads
document.addEventListener('DOMContentLoaded', function() {
    // Check if annotator name is stored in localStorage
    const storedAnnotator = localStorage.getItem('annotator');
    const annotatorInput = document.querySelector('input[name="annotator"]');
    
    // Always use localStorage value if it exists, regardless of template value
    if (storedAnnotator && annotatorInput) {
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
    
    // Restore auto-submit checkbox state
    const storedAutoSubmit = localStorage.getItem('auto-submit');
    const autoSubmitCheckbox = document.getElementById('auto-submit');
    
    if (autoSubmitCheckbox) {
        // Default is checked, so only uncheck if explicitly set to false
        if (storedAutoSubmit === 'false') {
            autoSubmitCheckbox.checked = false;
        }
        
        // Save auto-submit state when it changes
        autoSubmitCheckbox.addEventListener('change', function() {
            localStorage.setItem('auto-submit', this.checked);
        });
    }
    
    // Add event listeners for text inputs to check auto-submit
    const textInputs = document.querySelectorAll('.text-input');
    textInputs.forEach(input => {
        input.addEventListener('input', function() {
            checkAutoSubmit();
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