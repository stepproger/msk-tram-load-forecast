package ru.moscow.tram.forecast.api;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.reactive.function.client.WebClientRequestException;
import org.springframework.web.server.MissingRequestValueException;
import org.springframework.web.server.ResponseStatusException;
import org.springframework.web.server.ServerWebInputException;

@RestControllerAdvice
public class ApiErrorHandler {
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ApiError> badRequest(IllegalArgumentException exception) {
        return ResponseEntity.badRequest().body(new ApiError("invalid_parameter", exception.getMessage()));
    }

    @ExceptionHandler(ServerWebInputException.class)
    public ResponseEntity<ApiError> invalidInput(ServerWebInputException exception) {
        String parameter = exception.getMethodParameter() == null ? null : exception.getMethodParameter().getParameterName();
        String message = exception instanceof MissingRequestValueException missing
                ? "Не указан обязательный параметр " + missing.getName()
                : parameter == null ? "Некорректные параметры запроса" : "Некорректное значение параметра " + parameter;
        return ResponseEntity.badRequest().body(new ApiError("invalid_parameter", message));
    }

    @ExceptionHandler(ResponseStatusException.class)
    public ResponseEntity<ApiError> upstreamStatus(ResponseStatusException exception) {
        HttpStatus status = HttpStatus.valueOf(exception.getStatusCode().value());
        String code = switch (status) {
            case BAD_REQUEST, UNPROCESSABLE_CONTENT -> "invalid_parameter";
            case NOT_FOUND -> "not_found";
            case UNAUTHORIZED, FORBIDDEN -> "internal_auth_failed";
            default -> "upstream_error";
        };
        if (status == HttpStatus.UNPROCESSABLE_CONTENT) {
            status = HttpStatus.BAD_REQUEST;
        }
        if (code.equals("internal_auth_failed")) {
            // A rejected internal token is a deployment problem, not the caller's fault.
            return ResponseEntity.status(HttpStatus.BAD_GATEWAY)
                    .body(new ApiError(code, "Сервис данных отклонил межсервисный токен: проверьте INTERNAL_TOKEN"));
        }
        String message = exception.getReason() == null ? "Не удалось получить данные прогноза" : exception.getReason();
        return ResponseEntity.status(status).body(new ApiError(code, message));
    }

    @ExceptionHandler(WebClientRequestException.class)
    public ResponseEntity<ApiError> unavailable(WebClientRequestException exception) {
        return ResponseEntity.status(HttpStatus.SERVICE_UNAVAILABLE)
                .body(new ApiError("forecast_data_unavailable", "Сервис данных прогноза временно недоступен"));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiError> unexpected(Exception exception) {
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ApiError("internal_error", "Внутренняя ошибка сервиса: " + exception.getClass().getSimpleName()));
    }

    public record ApiError(String error, String message) {}
}
