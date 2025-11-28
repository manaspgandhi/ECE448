param(
    [string]$Path
)

$header = @"
{
    "frequency": {
        "pos": {
"@

$tempFile = [System.IO.Path]::GetTempFileName()
$writer = New-Object System.IO.StreamWriter($tempFile)
$writer.WriteLine($header)

$reader = New-Object System.IO.StreamReader($Path)
while (-not $reader.EndOfStream) {
    $writer.WriteLine($reader.ReadLine())
}

$reader.Close()
$writer.Close()

Move-Item -Path $tempFile -Destination $Path -Force
