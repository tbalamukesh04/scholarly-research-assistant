async function submitQuery() {
  const queryField = document.getElementById("query");
  const responseBox = document.getElementById("output");
  const submitBtn = document.getElementById("submit-btn")

  const queryValue = queryField.value.trim();
  if (!queryValue) return;
  responseBox.innerText = "Thinking....";
  
  submitBtn.disabled = true;
  submitBtn.innerText = "Processing...";
  responseBox.className = "";
  responseBox.innerText = "Thinking...";
  
  try {
    const res = await fetch("/query", {
      method: "POST", 
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query: queryValue,
        top_k: 10,
        mode: "strict"
      })
    });
  
    const data = await res.json();
    
    const isRefusal = (data.metrics && data.metrics.refused_count > 0) || 
                      (data.answer && data.answer.includes("cannot answer"));
    
    const isTruncated = data.metrics && data.metrics.truncated;
    
    responseBox.innerHTML = "";
    
    if (isRefusal) {
      responseBox.className = "refusal";
      responseBox.innerText = data.answer ?? "No Answer Found.";
    } else {
      responseBox.className = "success";
      
      const answerText = document.createTextNode(data.answer ?? "No Answer found.");
      responseBox.appendChild(answerText);
      
      if (isTruncated) {
        const warningDiv = document.createElement("div");
        warningDiv.className = "trunction-warning";
        warning.warningDiv.innerText = `[WARNING] Answer Truncated due to weak evidence. (${data.metrics.dropped_sentences}) sentences dropped`;
        responseBox.appendChild(warningDiv);
      }
    }
  } catch(err) {
    responseBox.className = "error";
    responseBox.innerText = "Error: " + err.message;
  }
  finally{
    submitBtn.disabled = false;
    submitBtn.innerText = "Submit";
  }
}
