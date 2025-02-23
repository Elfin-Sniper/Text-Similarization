document.getElementById('similarityForm').addEventListener('submit', async function(event) {
    event.preventDefault();

    const text1 = document.getElementById('text1').value;
    const text2 = document.getElementById('text2').value;

    const resultDiv = document.getElementById('result');
    resultDiv.textContent = 'Checking similarity...';

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text1, text2 })
        });

        const data = await response.json();
        resultDiv.textContent = data.result;
        resultDiv.className = 'result ' + (data.result === 'Similar' ? 'similar' : 'not-similar');
    } catch (error) {
        resultDiv.textContent = 'An error occurred. Please try again.';
        resultDiv.className = 'result not-similar';
    }
});