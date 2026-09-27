package ru.moscow.tram.forecast.api;

import java.net.URI;
import java.time.LocalDate;
import java.util.List;
import java.util.Map;
import org.springframework.http.ResponseEntity;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.HttpStatusCode;
import org.springframework.stereotype.Component;
import org.springframework.web.reactive.function.client.ClientResponse;
import org.springframework.web.reactive.function.client.WebClient;
import org.springframework.web.server.ResponseStatusException;
import reactor.core.publisher.Mono;

@Component
public class ForecastDataClient {
    private static final ParameterizedTypeReference<List<ReplayEvent>> REPLAY_EVENTS =
            new ParameterizedTypeReference<>() {};

    static final String INTERNAL_TOKEN_HEADER = "X-Internal-Token";
    private static final int MAX_RESPONSE_BYTES = 64 * 1024 * 1024;

    private final WebClient client;
    private final String baseUrl;

    public ForecastDataClient(
            WebClient.Builder builder,
            @Value("${forecast-data.base-url:http://localhost:8001}") String baseUrl,
            @Value("${forecast-data.internal-token}") String internalToken) {
        if (internalToken.isBlank()) {
            throw new IllegalStateException("INTERNAL_TOKEN должен быть задан");
        }
        this.baseUrl = baseUrl.replaceAll("/+$", "");
        // Full-period exports and yearly hourly series exceed the 256 KB default codec buffer.
        this.client = builder.baseUrl(baseUrl)
                .defaultHeader(INTERNAL_TOKEN_HEADER, internalToken)
                .codecs(codecs -> codecs.defaultCodecs().maxInMemorySize(MAX_RESPONSE_BYTES))
                .build();
    }

    private static Mono<ResponseStatusException> upstreamError(ClientResponse response) {
        return response.bodyToMono(InternalError.class)
                .onErrorResume(error -> Mono.empty())
                .filter(error -> error.detail() != null && !error.detail().isBlank())
                .defaultIfEmpty(new InternalError("Внутренний сервис прогноза недоступен"))
                .map(error -> new ResponseStatusException(response.statusCode(), error.detail()));
    }

    /** Proxies a data-service response as raw bytes: no JSON re-serialization on the hot path. */
    public Mono<ResponseEntity<byte[]>> getJson(String path, String rawQuery) {
        String target = baseUrl + path + (rawQuery == null || rawQuery.isEmpty() ? "" : "?" + rawQuery);
        return client.get().uri(URI.create(target)).exchangeToMono(ForecastDataClient::passThrough);
    }

    public Mono<ResponseEntity<byte[]>> send(HttpMethod method, String path, byte[] body) {
        return client.method(method).uri(path)
                .contentType(MediaType.APPLICATION_JSON)
                .bodyValue(body == null ? new byte[0] : body)
                .exchangeToMono(ForecastDataClient::passThrough);
    }

    private static Mono<ResponseEntity<byte[]>> passThrough(ClientResponse response) {
        if (response.statusCode().isError()) {
            return upstreamError(response).flatMap(Mono::error);
        }
        return response.bodyToMono(byte[].class).defaultIfEmpty(new byte[0])
                .map(bytes -> ResponseEntity.status(response.statusCode())
                        .headers(headers -> {
                            response.headers().contentType().ifPresent(headers::setContentType);
                            response.headers().header(HttpHeaders.CONTENT_DISPOSITION).stream().findFirst()
                                    .ifPresent(value -> headers.set(HttpHeaders.CONTENT_DISPOSITION, value));
                        })
                        .body(bytes));
    }

    public Mono<List<ReplayEvent>> replayEvents(LocalDate date) {
        return client.get()
                .uri(uri -> uri.path("/internal/replay-events").queryParam("date", date).build())
                .retrieve()
                .onStatus(HttpStatusCode::isError, ForecastDataClient::upstreamError)
                .bodyToMono(REPLAY_EVENTS);
    }

    public Mono<NowcastInputs> nowcastInputs(int route, LocalDate date) {
        return client.get()
                .uri(uri -> uri.path("/internal/nowcast-inputs")
                        .queryParam("route", route)
                        .queryParam("date", date)
                        .build())
                .retrieve()
                .onStatus(HttpStatusCode::isError, ForecastDataClient::upstreamError)
                .bodyToMono(NowcastInputs.class);
    }

    public Mono<InternalHealth> health() {
        return client.get()
                .uri("/internal/health")
                .retrieve()
                .bodyToMono(InternalHealth.class);
    }

    public record ReplayEvent(String ts, int route, int hour, int boardings_so_far) {}
    public record ActualHour(int hour, int boardings) {}
    public record BaselineHour(int hour, double yhat, double q10_ratio, double q90_ratio) {}
    public record NowcastConfig(double shrink, List<Double> clip, int start_hour, String gain_note) {}
    public record NowcastInputs(
            int route,
            LocalDate date,
            List<ActualHour> actuals,
            List<BaselineHour> baseline,
            NowcastConfig nowcast) {}
    public record InternalHealth(String status, String model_version, Double lb_wape_score,
                                 Double cv_wape_score, String trained_at) {}
    private record InternalError(String detail) {}
}
