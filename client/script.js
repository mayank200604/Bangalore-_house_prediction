// Load locations on page load
window.addEventListener('DOMContentLoaded', () => {
  fetch('/locations')
    .then(response => response.json())
    .then(data => {
      const select = document.getElementById('location');
      select.innerHTML = '<option disabled selected>Select Location</option>';
      data.locations.forEach(loc => {
        const option = document.createElement('option');
        option.value = loc;
        option.textContent = loc;
        select.appendChild(option);
      });
    })
    .catch(err => {
      console.error('Error loading locations:', err);
      const select = document.getElementById('location');
      select.innerHTML = '<option disabled>Error loading</option>';
    });
});

// Handle prediction button click
document.getElementById('estimate-btn').addEventListener('click', function () {
  const sqft = parseFloat(document.getElementById('sqft').value);
  const bhk = parseInt(document.querySelector('input[name="bhk"]:checked')?.value || 0);
  const bath = parseInt(document.querySelector('input[name="bath"]:checked')?.value || 0);
  const location = document.getElementById('location').value;

  if (!sqft || !bhk || !bath || !location) {
    alert('Please fill in all fields');
    return;
  }

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      total_squareft: sqft,
      bhk: bhk,
      bath: bath,
      location: location
    })
  })
    .then(response => response.json())
    .then(data => {
      document.getElementById('result').innerText =
        `Predicted Price: ₹ ${data.predicted_price_lakhs} lakhs`;
    })
    .catch(err => {
      document.getElementById('result').innerText = 'Error fetching price.';
      console.error(err);
    });
});
