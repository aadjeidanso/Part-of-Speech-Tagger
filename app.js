document.getElementById('tag-button').addEventListener('click', function() {
    const sentence = document.getElementById('input-sentence').value;
    if (!sentence) return alert('Please enter a sentence');

    fetch('http://127.0.0.1:8080/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sentence: sentence })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        document.getElementById('output').innerHTML = `Sentence: ${data.sentence}<br>POS Tags: ${data.pos_tags.join(', ')}`;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred. Please check the console for details.');
    });
});
