document.addEventListener('DOMContentLoaded', () => {
    const inputs = ['resume', 'jd', 'transcript'];
    
    inputs.forEach(id => {
        const inputElement = document.getElementById(id);
        const nameElement = document.getElementById(`${id}-name`);
        
        inputElement.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                nameElement.textContent = e.target.files[0].name;
                nameElement.style.color = '#000000'; // Darken text when file selected
            } else {
                nameElement.textContent = 'No file chosen';
                nameElement.style.color = 'var(--gray-text)';
            }
        });
    });
    const form = document.getElementById('screener-form');
    const submitBtn = document.getElementById('submit-btn');
    const btnText = document.querySelector('.btn-text');
    const loader = document.querySelector('.loader');
    
    const resultContainer = document.getElementById('result-container');
    const decisionText = document.getElementById('decision-text');
    const modelUsedText = document.getElementById('model-used-text');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        submitBtn.disabled = true;
        btnText.classList.add('hidden');
        loader.classList.remove('hidden');
        resultContainer.classList.add('hidden');
        const formData = new FormData();
        formData.append('role', document.getElementById('role').value);
        formData.append('resume', document.getElementById('resume').files[0]);
        formData.append('jd', document.getElementById('jd').files[0]);
        
        const transcriptFile = document.getElementById('transcript').files[0];
        if (transcriptFile) {
            formData.append('transcript', transcriptFile);
        }

        try {
            const response = await fetch('http://localhost:8000/predict', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`Server responded with status: ${response.status}`);
            }

            const data = await response.json();
            
            decisionText.textContent = data.decision;
            decisionText.className = ''; // Reset classes
            decisionText.classList.add(data.decision.toLowerCase());
            
            modelUsedText.textContent = `Generated using: ${data.model_used}`;
            
            resultContainer.classList.remove('hidden');
            
        } catch (error) {
            console.error('Error screening candidate:', error);
            alert('Failed to screen candidate. Make sure your FastAPI backend is running on http://localhost:8000');
        } finally {
            // Restore UI state
            submitBtn.disabled = false;
            btnText.classList.remove('hidden');
            loader.classList.add('hidden');
        }
    });
});
