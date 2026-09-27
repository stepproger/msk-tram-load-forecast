package ru.moscow.tram.forecast.api;

public record HealthResponse(String status, String model_version, Double lb_wape_score,
                             Double cv_wape_score, String trained_at) {
}
