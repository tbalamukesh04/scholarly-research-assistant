async function submitQuery() {
  const queryValue = document.getElementById("query").value;
  const responseBox = document.getElementById("output");

  responseBox.innerText = "Thinking....";
  
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
    responseBox.innerText = data.answer ?? "No answer found.";
  } catch(err) {
    responseBox.innerText = "Error: " + err.message;
  }
}

// async function submitQuery() {
//   const query = document.getElementById("query").value;
//   const responseBox = document.getElementById("response");

//   const payload = {
//     query: query,
//     top_k: 10,
//     mode: "strict"
//   };
  
//   responseBox.innerText = "Thinking....";
  
//   try {const res = await fetch("/query", {
//       method: "POST",
//       headers: { "Content-Type": "application/json" },
//       body: JSON.stringify({
//         query: query,
//         top_k: 10,
//         mode: "strict"
//       })
//     });
  
//     const data = await res.json();
//     responseBox.innerText = data.answer;
// } catch(err) {
//     responseBox.innerText = "Error: " + err.message;
// }

// }