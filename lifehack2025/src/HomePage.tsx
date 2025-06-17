import React, { useState, useRef } from 'react';
import type { ChangeEvent, FC, DragEvent } from 'react';

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

    const handlePdfUpload = (e: ChangeEvent<HTMLInputElement>): void => {
        if (e.target.files) {
            handleFile(e.target.files[0]);
        }
    }

    const handleDrop = (e: DragEvent<HTMLDivElement>): void => {
        e.preventDefault();
        setIsDragging(false);

        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    }

    const handleSubmit = () => {
        // Submit logic here
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
        </div>
    );
}

export default HomePage;