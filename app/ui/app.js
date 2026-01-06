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
    if (isRefusal) {
      responseBox.className = "refusal";
    } else {
      responseBox.className = "success";
    }
    
    responseBox.innerText = data.answer ?? "No answer found.";
  } catch(err) {
    responseBox.className = "error";
    responseBox.innerText = "Error: " + err.message;
  }
  finally{
    submitBtn.disabled = false;
    submitBtn.innerText = "Submit";
  }
}

// async function submitQuery() {
//   const queryField = document.getElementById("query");
//   const responseBox = document.getElementById("output");
//   const submitBtn = document.getElementById("submit-btn");

//   const queryValue = queryField.value.trim();
//   if (!queryValue) return;

//   // 1. Lock UI
//   submitBtn.disabled = true;
//   submitBtn.innerText = "Processing...";
//   responseBox.className = ""; // Reset classes
//   responseBox.innerText = "Thinking...";
  
//   try {
//     const res = await fetch("/query", {
//       method: "POST", 
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({
//         query: queryValue,
//         top_k: 10,
//         mode: "strict"
//       })
//     });
  
//     const data = await res.json();

//     // 2. Determine State based on Metrics or Content
//     const isRefusal = (data.metrics && data.metrics.refused_count > 0) || 
//                       (data.answer && data.answer.includes("cannot answer"));

//     if (isRefusal) {
//         responseBox.className = "refusal";
//     } else {
//         responseBox.className = "success";
//     }

//     responseBox.innerText = data.answer ?? "No answer found.";

//   } catch(err) {
//     responseBox.className = "error";
//     responseBox.innerText = "Error: " + err.message;
//   } finally {
//     // 3. Unlock UI
//     submitBtn.disabled = false;
//     submitBtn.innerText = "Submit";
//   }
// }