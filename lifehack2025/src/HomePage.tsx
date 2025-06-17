import React, { useState, useRef } from 'react';
import type { ChangeEvent, FC, DragEvent } from 'react';
import axios from 'axios';

type ColorBlindType = "none" | "protanopia" | "deuteranopia" | "tritanopia"

    const HomePage: FC = (): React.JSX.Element => {
    // States
    const [colorBlindType, setColorBlindType] = useState<ColorBlindType>("none");
    const [isColorBlind, setIsColorBlind] = useState<boolean>(false);
    const [isSpeechEnabled, setIsSpeechEnabled] = useState<boolean>(false);
    const [pdfFile, setPdfFile] = useState<File | null>(null);
    const [isDragging, setIsDragging] = useState<boolean>(false);
    const [fileInputKey, setFileInputKey] = useState<number>(Date.now());
    const fileInputRef = useRef<HTMLInputElement>(null);

    const handleColorBlindType = (e: React.ChangeEvent<HTMLSelectElement>) => {
        setColorBlindType(e.target.value as ColorBlindType);
    }

    const handleIsColorBlind = (): void => {
        setIsColorBlind(prev => {
            if (prev) {
                setColorBlindType("none");
            }
            return !prev;
        });
    }

    const handleIsSpeechEnabled = (): void => {
        setIsSpeechEnabled(prev => !prev);
    }

    const handleFile = (file: File): void => {
        if (file.type === "application/pdf") {
            setPdfFile(file);
            setFileInputKey(Date.now());
            alert("Upload successful")
        } else {
            alert("Please upload a valid PDF file.");
        }
    }

    const handlePdfUpload = async (e: ChangeEvent<HTMLInputElement>): Promise<void> => {
        if (e.target.files) {
            handleFile(e.target.files[0]);

            const formData = new FormData();
            formData.append("file", e.target.files[0]);

            try {
                const response = await axios.post("http://localhost:8000/upload/", formData, {
                    headers: { "Content-Type": "multipart/form-data" }
                });
                    console.log(response.data);
            } catch (error) {
                console.error(error);
            }
        }
    }

    const handleDrop = (e: DragEvent<HTMLDivElement>): void => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    }

    const handleSubmit = async () => {
        // Submit logic here
        try {
            const response = await axios.post('http://localhost:8000/convert/', {
            toSpeech: isSpeechEnabled,
            toColour: isColorBlind
        });
        console.log(response.data);
        } catch (error) {
            console.error(error);
        }
    }

    const handleDownload = async () => {
        try {
            const response = await fetch('http://localhost:8000/download/', {
            method: 'GET',
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'converted_files.zip';
        document.body.appendChild(a);
        a.click();
        a.remove();
        window.URL.revokeObjectURL(url);
        } catch (error) {
            console.error('Download failed:', error);
        }
    }

    return (
        <div className="container">
            <h1 className="main-title">Document Accessibility Converter</h1>
            
            <div className="settings-panel">
                {/* Checkbox for audio support*/}
                <div className="checkbox-container">
                    <label className="checkbox-label">
                        <input
                            type="checkbox"
                            checked={isSpeechEnabled}
                            onChange={handleIsSpeechEnabled}
                            className="checkbox-input"
                        />
                        <span className="checkbox-custom"></span>
                        Enable Speech Support
                    </label>
                </div>
                
                {/* Checkbox for color blind */}
                <div className="checkbox-container">
                    <label className="checkbox-label">
                        <input
                            type="checkbox"
                            checked={isColorBlind}
                            onChange={handleIsColorBlind}
                            className="checkbox-input"
                        />
                        <span className="checkbox-custom"></span>
                        Enable Color Blind Support
                    </label>
                </div>
                
                {/* Dropdown for ColorBlindness */}
                {isColorBlind && (
                    <div className="dropdown-container">
                        <label className="dropdown-label">
                            Type of Color Blindness:
                        </label>
                        <select
                            value={colorBlindType}
                            onChange={(e) => handleColorBlindType(e)}
                            className="dropdown-select"
                        >
                            <option disabled value="">-- Select --</option>
                            <option value="none">None</option>
                            <option value="protanopia">Protanopia (Red-Blind)</option>
                            <option value="deuteranopia">Deuteranopia (Green-Blind)</option>
                            <option value="tritanopia">Tritanopia (Blue-Blind)</option>
                        </select>
                    </div>
                )}
            </div>

            {/* pdf upload */}
            <div className="upload-section">
                <h3 className="upload-title">
                    Upload PDF File
                </h3>
                <div
                    className={`upload-area ${isDragging ? 'dragging' : ''}`}
                    onDrop={handleDrop}
                    onDragOver={(e) => {
                        e.preventDefault();
                        setIsDragging(true);
                    }}
                    onDragLeave={() => setIsDragging(false)}
                >
                    <div className="upload-content">
                        <svg className="upload-icon" viewBox="0 0 24 24">
                            <path d="M19,13H13V19H11V13H5V11H11V5H13V11H19V13Z" />
                        </svg>
                        <button
                            className="upload-button"
                            onClick={() => fileInputRef.current?.click()}
                        >
                            Select PDF
                        </button>
                        <p className="upload-text">or drag and drop a PDF in the box</p>
                        <input
                            key={fileInputKey}
                            type="file"
                            accept="application/pdf"
                            onChange={handlePdfUpload}
                            ref={fileInputRef}
                            className="upload-input"
                        />
                    </div>
                </div>
                
                <div className="file-info">
                    {pdfFile ? (
                        <p className="file-name">
                            <span className="file-icon">📄</span> {pdfFile.name}
                        </p>
                    ) : (
                        <p className="no-file">No file selected</p>
                    )}
                </div>
            </div>

            <button
                className="submit-button"
                onClick={handleSubmit}
                disabled={!pdfFile}
            >
                Convert Document
            </button>

            {/* Temporary download button */}
            <button
                onClick={handleDownload}
            >
                Download
            </button>
        </div>
    );
}

export default HomePage;