#!/usr/bin/perl
package AI::ANN::Neuron;
BEGIN {
  $AI::ANN::Neuron::VERSION = '0.005';
}
# ABSTRACT: a neuron for an artificial neural network simulator

use strict;
use warnings;

use 5.014_000;
use Moose;


has 'id' => (is => 'rw', isa => 'Int');
has 'inputs' => (is => 'rw', isa => 'HashRef[Num]', required => 1);
has 'neurons' => (is => 'rw', isa => 'HashRef[Num]', required => 1);

around BUILDARGS => sub {
	my $orig = shift;
	my $class = shift;
	my %data;
	if ( @_ == 2 && ref $_[0] eq 'HASH' && ref $_[1] eq 'HASH' ) {
		%data = ('inputs' => $_[0], 'neurons' => $_[1]);
	} elsif ( @_ == 3 && ref $_[1] eq 'HASH' && ref $_[2] eq 'HASH' ) {
		%data = ('id' => $_[0], 'inputs' => $_[1], 'neurons' => $_[2]);
	} elsif ( @_ == 1 && ref $_[0] eq 'HASH' ) {
		%data = %{$_[0]};
	} else {
		%data = @_;
	}
	foreach my $i (keys %{$data{'inputs'}}) {
		unless (defined $data{'inputs'}->{$i} && $data{'inputs'}->{$i} > 0) {
			delete $data{'inputs'}->{$i};
		}
	}
	foreach my $i (keys %{$data{'neurons'}}) {
		unless (defined $data{'neurons'}->{$i} && $data{'neurons'}->{$i} > 0) {
			delete $data{'neurons'}->{$i};
		}
	}
	return $class->$orig(%data);
};


sub ready {
	my $self = shift;
	my $inputs = shift;
	my $neurons = shift;

	foreach my $id (keys $self->{'inputs'}) { # Requires Perl 5.14 !!!
		unless ((not defined $self->{'inputs'}->{$id}) || 
				$self->{'inputs'}->{$id} == 0 || defined $inputs->[$id])
				{return 0}
		# This probably shouldn't ever happen, as it would be weird if our
		# inputs weren't available yet.
	}
	foreach my $id (keys $self->{'neurons'}) {
		unless ((not defined $self->{'neurons'}->{$id}) || 
				$self->{'neurons'}->{$id} == 0 || defined $neurons->{$id})
				{return 0}
		# First clause should be redundant, but you can never be too safe
	}
	return 1;
}


sub execute {
	my $self = shift;
	my $inputs = shift;
	my $neurons = shift;
	my $output = 0;
	foreach my $id (keys $self->{'inputs'}) {
		$output += ($self->{'inputs'}->{$id} || 0 ) * ($inputs->[$id] || 0);
	}
	foreach my $id (keys $self->{'neurons'}) {
		$output += ($self->{'neurons'}->{$id} || 0) * ($neurons->{$id} || 0);
	}
	return $output;
}

1;
		

__END__
=pod

=head1 NAME

AI::ANN::Neuron - a neuron for an artificial neural network simulator

=head1 VERSION

version 0.005

=head1 METHODS

=head2 new

AI::ANN::Neuron->new( $neuronid, {$inputid => $weight, ...}, {$neuronid => $weight} )

Weights may be whatever the user chooses. Note that packages that use this 
one may place their own restructions. Neurons and inputs are assumed to be 
zero-indexed.

=head2 ready

$neuron->ready( [$input0, $input1, ...], {$neuronid => $neuronvalue, ...} )

All inputs must be provided or you're insane.
Returns 1 if ready, 0 otherwise.

=head2 execute

$neuron->execute( [$input0, $input1, ...], {$neuronid => $neuronvalue, ...} )

All inputs must be provided or you're insane
Returns raw value (linear potential)

=head1 AUTHOR

Dan Collins <DCOLLINS@cpan.org>

=head1 COPYRIGHT AND LICENSE

This software is Copyright (c) 2011 by Dan Collins.

This is free software, licensed under:

  The GNU General Public License, Version 3, June 2007

=cut

