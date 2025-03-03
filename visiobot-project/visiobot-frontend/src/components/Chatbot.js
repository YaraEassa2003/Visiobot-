import React, { useState } from "react";
import axios from "axios";

const Chatbot = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [selectedFile, setSelectedFile] = useState(null);
  const [datasetInfo, setDatasetInfo] = useState(null);
  const [recommendedChart, setRecommendedChart] = useState("");

  // Handle User Messages
  const sendMessage = async () => {
    if (!input) return;
    
    const userMessage = { role: "user", content: input };
    setMessages([...messages, userMessage]);

    try {
      const response = await axios.post("http://127.0.0.1:5000/chat", {
        message: input,
      });

      const botMessage = { role: "bot", content: response.data.response };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error("Error:", error);
    }

    setInput("");
  };

  // Handle File Selection
  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  // Handle File Upload
  const uploadFile = async () => {
    if (!selectedFile) {
      alert("Please select a file first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/process-dataset", formData, {
        headers: { "Content-Type": "multipart/form-data" }
      });

      setDatasetInfo(response.data);
      const botMessage = { role: "bot", content: "Dataset uploaded! Now please provide the purpose and target audience." };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error("Upload Error:", error);
    }
  };

  // Send Dataset Info to the Model for Prediction
  const getVisualization = async () => {
    if (!datasetInfo) {
      alert("Please upload a dataset first.");
      return;
    }

    try {
      const response = await axios.post("http://127.0.0.1:5000/get-visualization", datasetInfo);
      setRecommendedChart(response.data.recommended_visualization);
      const botMessage = { role: "bot", content: `The recommended visualization is: ${response.data.recommended_visualization}` };
      setMessages((prevMessages) => [...prevMessages, botMessage]);
    } catch (error) {
      console.error("Prediction Error:", error);
    }
  };

  return (
    <div>
      <h2>Welcome to VisioBot</h2>
      <h3>VisioBot Chat</h3>
      <div style={{ height: "300px", overflowY: "scroll", border: "1px solid black", padding: "10px" }}>
        {messages.map((msg, index) => (
          <p key={index} style={{ color: msg.role === "user" ? "blue" : "green" }}>
            <strong>{msg.role === "user" ? "You" : "VisioBot"}:</strong> {msg.content}
          </p>
        ))}
      </div>

      <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask something..." />
      <button onClick={sendMessage}>Send</button>

      <div style={{ marginTop: "10px" }}>
        <input type="file" onChange={handleFileChange} />
        <button onClick={uploadFile}>Upload Dataset</button>
      </div>

      {datasetInfo && (
        <button onClick={getVisualization} style={{ marginTop: "10px" }}>Get Visualization</button>
      )}

      {recommendedChart && <h3>Recommended Visualization: {recommendedChart}</h3>}
    </div>
  );
};

export default Chatbot;
