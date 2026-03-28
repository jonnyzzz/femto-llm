FROM golang:1.25-alpine AS builder

WORKDIR /app
COPY go.mod go.sum* ./
RUN go mod download 2>/dev/null || true
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o femtollm ./cmd/femtollm

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/femtollm /usr/local/bin/femtollm

EXPOSE 8080
CMD ["femtollm"]
