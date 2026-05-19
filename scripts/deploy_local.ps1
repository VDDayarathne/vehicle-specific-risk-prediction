param(
  [switch]$Build,
  [switch]$Detach
)

$compose = "docker-compose.yml"
$args = @("-f", $compose, "up")
if ($Build) { $args += "--build" }
if ($Detach) { $args += "-d" }

docker compose @args
