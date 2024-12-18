document.getElementById("classify-btn").addEventListener("click", async () => {
  const text = document.getElementById("input-text").value.trim();
  if (!text) return;

  const resp = await fetch("/classify", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  const { label, probability } = await resp.json();
  document.getElementById(
    "result"
  ).innerText = `Résultat : ${label} (probabilité : ${probability.toFixed(4)})`;
});
