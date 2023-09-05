#version 450

layout( binding = 0 ) restrict readonly buffer BufferPoints {
	vec2 inPoints[];
};

void main()
{
	gl_Position = vec4( inPoints[gl_VertexIndex], 0.0, 1.0 );
	gl_PointSize = 1.0;
}