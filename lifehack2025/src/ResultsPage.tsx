import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import type { FC } from 'react';

interface ResultData {
    originalName: string;
    convertedUrl: string;
}

const ResultPage: FC = () => {
    const location = useLocation();
    const navigate = useNavigate();

    return(
        <>
            
        </>
    );
}